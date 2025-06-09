import os
import re
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import toml
import boto3
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ECRImage:
    repository: str
    tag: str
    pushed_at: datetime
    image_uri: str
    compute_type: str = ""
    cuda_version: str = ""
    pytorch_version: str = ""

class ECRImageSelector:
    def __init__(self, account_id: str = "763104351884", region: str = "us-west-2"):
        self.account_id = account_id
        self.region = region
        try:
            self.ecr_client = boto3.client('ecr', region_name=region)
            self.ecr_client.describe_repositories(maxResults=1)
            self.logger = logging.getLogger(__name__)
            self.logger.info("✅ AWS ECR credentials verified")
        except Exception as e:
            print(f"❌ ECR access failed: {e}")
            print("💡 Please configure your AWS credentials:")
            raise Exception("AWS credentials required to access ECR")
    
    def get_pytorch_images(self, framework_type: str) -> List[ECRImage]:
        """Get pytorch-training or pytorch-inference images"""
        repository = f"pytorch-{framework_type}"
        try:
            paginator = self.ecr_client.get_paginator('describe_images')
            images = []
            for page in paginator.paginate(
                registryId=self.account_id,
                repositoryName=repository
            ):
                for image_detail in page['imageDetails']:
                    if 'imageTags' not in image_detail:
                        continue
                    for tag in image_detail['imageTags']:
                        if 'sagemaker' not in tag:
                            continue
                        ecr_image = ECRImage(
                            repository=repository,
                            tag=tag,
                            pushed_at=image_detail['imagePushedAt'],
                            image_uri=f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repository}:{tag}"
                        )
                        self.parse_image(ecr_image)
                        images.append(ecr_image)
            images.sort(key=lambda x: x.pushed_at, reverse=True)
            return images
        except Exception as e:
            print(f"❌ Error getting {repository} images: {e}")
            print("💡 Make sure your AWS credentials have ECR read permissions")
            return []
    
    def parse_image(self, image: ECRImage):
        """Parse PyTorch version, CUDA version and compute type from tag"""
        tag = image.tag            
        if '-gpu-' in tag:
            image.compute_type = 'gpu'
        elif '-cpu-' in tag:
            image.compute_type = 'cpu'
        cuda_match = re.search(r'cu(\d+)', tag)
        if cuda_match:
            image.cuda_version = f"cu{cuda_match.group(1)}"
        else:
            image.cuda_version = ""  
        pytorch_match = re.match(r'^(\d+\.\d+\.\d+)', tag)
        if pytorch_match:
            image.pytorch_version = pytorch_match.group(1)
        else:
            image.pytorch_version = ""
    
    def select_matching_cuda_images(self) -> Dict:
        """
        Select 4 images: 2 CPU (no CUDA) + 2 GPU (matching CUDA)
        Prioritize same PyTorch version and highest CUDA version
        """
        training_images = self.get_pytorch_images("training")
        inference_images = self.get_pytorch_images("inference")
        if not training_images or not inference_images:
            print("❌ Could not retrieve images from ECR")
            return None
        training_cpu = [img for img in training_images if img.compute_type == 'cpu']
        training_gpu = [img for img in training_images if img.compute_type == 'gpu']
        inference_cpu = [img for img in inference_images if img.compute_type == 'cpu']
        inference_gpu = [img for img in inference_images if img.compute_type == 'gpu']
        if not training_cpu or not inference_cpu:
            print("❌ Missing CPU images")
            return None
        combinations = defaultdict(lambda: {'training_cpu': [], 'training_gpu': [], 'inference_cpu': [], 'inference_gpu': []})
        for img in training_cpu:
            if hasattr(img, 'pytorch_version') and img.pytorch_version:
                combinations[img.pytorch_version]['training_cpu'].append(img)
        for img in inference_cpu:
            if hasattr(img, 'pytorch_version') and img.pytorch_version:
                combinations[img.pytorch_version]['inference_cpu'].append(img)
        for img in training_gpu:
            if hasattr(img, 'pytorch_version') and img.pytorch_version and img.cuda_version:
                key = f"{img.pytorch_version}+{img.cuda_version}"
                combinations[key]['training_gpu'].append(img)
        for img in inference_gpu:
            if hasattr(img, 'pytorch_version') and img.pytorch_version and img.cuda_version:
                key = f"{img.pytorch_version}+{img.cuda_version}"
                combinations[key]['inference_gpu'].append(img)
        best_selection = None
        pytorch_versions = set()
        for key in combinations.keys():
            if '+' not in key:  
                pytorch_versions.add(key)
            else:  
                pytorch_versions.add(key.split('+')[0])
        pytorch_versions = sorted(pytorch_versions, key=lambda x: [int(i) for i in x.split('.')], reverse=True)
        print(f"📋 Available PyTorch versions: {pytorch_versions}")
        for pytorch_version in pytorch_versions:
            if pytorch_version not in combinations:
                continue
            if not combinations[pytorch_version]['training_cpu'] or not combinations[pytorch_version]['inference_cpu']:
                continue
            cuda_versions = []
            for key in combinations.keys():
                if key.startswith(f"{pytorch_version}+"):
                    cuda_version = key.split('+')[1]
                    if (combinations[key]['training_gpu'] and combinations[key]['inference_gpu']):
                        cuda_versions.append(cuda_version)
            if cuda_versions:
                cuda_versions.sort(key=lambda x: int(x[2:]), reverse=True)
                best_cuda = cuda_versions[0]
                gpu_key = f"{pytorch_version}+{best_cuda}"
                best_selection = {
                    'pytorch_version': pytorch_version,
                    'cuda_version': best_cuda,
                    'training_cpu': combinations[pytorch_version]['training_cpu'][0],      
                    'training_gpu': combinations[gpu_key]['training_gpu'][0],              
                    'inference_cpu': combinations[pytorch_version]['inference_cpu'][0],    
                    'inference_gpu': combinations[gpu_key]['inference_gpu'][0]             
                }
                print(f"✅ Found complete set with PyTorch {pytorch_version} and CUDA {best_cuda}")
                break
        if not best_selection:
            print("❌ Could not find a complete set with matching PyTorch and CUDA versions")
            return None
        return best_selection

class BaseAutomation:
    """Base class with common functionality"""
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        self.current_version = current_version
        self.previous_version = previous_version
        self.fork_url = fork_url
        current_parts = current_version.split('.')
        previous_parts = previous_version.split('.')
        self.is_major_release = (current_parts[0] != previous_parts[0] or 
                               (current_parts[1] != previous_parts[1] and current_parts[2] == '0'))
        self.short_version = current_version.replace('.', '')
        current_dir = Path(os.getcwd())
        self.main_project_dir = current_dir  
        self.workspace_dir = current_dir.parent / "deep-learning-container"
        self.repo_dir = self.workspace_dir / "deep-learning-containers"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Main project directory: {self.main_project_dir}")
        self.logger.info(f"Workspace directory: {self.workspace_dir}")
        self.logger.info(f"Repository directory: {self.repo_dir}")