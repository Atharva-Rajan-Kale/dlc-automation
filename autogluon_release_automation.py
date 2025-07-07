import os
import re
import sys
import yaml
import logging
import subprocess
import boto3
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
from common import BaseAutomation, ECRImageSelector
from github_pr_automation import GitHubPRAutomation
class AutoGluonReleaseImagesAutomation(BaseAutomation):
    """
    Automation for updating release_images files and available_images.md with AutoGluon configuration
    """
    PRODUCTION_ACCOUNT_ID = '763104351884'
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.training_file = self.repo_dir / "release_images_training.yml"
        self.inference_file = self.repo_dir / "release_images_inference.yml"
        self.available_images_file = self.repo_dir / "available_images.md"
        self.pr_automation = GitHubPRAutomation(
            current_version=current_version,
            fork_url=fork_url,
            repo_dir=self.repo_dir
        )
        self.original_training_content = None
        self.original_inference_content = None
        self.original_available_images_content = None

    def prompt_user(self, question: str) -> bool:
        """Prompt user with a yes/no question and repeat until valid answer"""
        while True:
            response = input(f"{question} (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
                
    def wait_for_cr_completion(self):
        """Wait for CR process completion"""
        print("🔍 Checking CR process completion status...")
        while True:
            if self.prompt_user("Is the CR process completed?"):
                print("✅ CR process confirmed as completed. Proceeding...")
                break
            else:
                print("⏳ Waiting for CR process completion...")
                
    def get_latest_autogluon_images_by_type(self, repo_name: str, account_id: str = None) -> Dict[str, Dict]:
        """Get the latest GPU and CPU images from autogluon repository"""
        if account_id is None:
            account_id = os.environ.get('ACCOUNT_ID')
        self.logger.info(f"🔍 Getting latest GPU and CPU images from {repo_name} using account {account_id}...")
        region = os.environ.get('REGION', 'us-east-1')
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set and no account_id provided")
        ecr_client = boto3.client('ecr', region_name=region)
        try:
            response = ecr_client.describe_images(
                registryId=account_id,
                repositoryName=repo_name,
                maxResults=50
            )
            images = sorted(
                response['imageDetails'], 
                key=lambda x: x['imagePushedAt'], 
                reverse=True
            )
            self.logger.info(f"📦 Found {len(images)} total images in {repo_name}")
            latest_gpu = None
            latest_cpu = None
            for image in images:
                if 'imageTags' in image:
                    tag = image['imageTags'][0]
                    if not latest_gpu and '-gpu-' in tag:
                        image_info = self.parse_image_tag(tag, require_cuda=True)
                        if image_info:
                            image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}"
                            image_info['image_uri'] = image_uri
                            image_info['tag'] = tag
                            latest_gpu = image_info
                            self.logger.info(f"📦 Found latest GPU image: {tag}")
                    if not latest_cpu and '-cpu-' in tag:
                        image_info = self.parse_image_tag(tag, require_cuda=False)
                        if image_info:
                            image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}"
                            image_info['image_uri'] = image_uri
                            image_info['tag'] = tag
                            latest_cpu = image_info
                            self.logger.info(f"📦 Found latest CPU image: {tag}")
                    if latest_gpu and latest_cpu:
                        break
            result = {}
            if latest_gpu:
                result['gpu'] = latest_gpu
            if latest_cpu:
                result['cpu'] = latest_cpu
            self.logger.info(f"✅ Found {len(result)} image types from {repo_name}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Failed to get images from {repo_name}: {e}")
            return {}
        
    def get_latest_autogluon_images(self, repo_name: str, count: int = 2, account_id: str = None, gpu_only: bool = False) -> list[Dict]:
        """Get the latest N images from any autogluon repository (beta or production) - Legacy method"""
        if account_id is None:
            account_id = os.environ.get('ACCOUNT_ID')
        self.logger.info(f"🔍 Getting latest {count} images from {repo_name} using account {account_id}...")
        region = os.environ.get('REGION', 'us-east-1')
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set and no account_id provided")
        ecr_client = boto3.client('ecr', region_name=region)
        try:
            response = ecr_client.describe_images(
                registryId=account_id,
                repositoryName=repo_name,
                maxResults=50
            )
            images = sorted(
                response['imageDetails'], 
                key=lambda x: x['imagePushedAt'], 
                reverse=True
            )
            self.logger.info(f"📦 Found {len(images)} total images in {repo_name}")
            result_images = []
            for image in images:
                if 'imageTags' in image and len(result_images) < count:
                    tag = image['imageTags'][0]
                    if gpu_only and '-cpu-' in tag:
                        continue
                    require_cuda = '-gpu-' in tag
                    image_info = self.parse_image_tag(tag, require_cuda=require_cuda)
                    if image_info:
                        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}"
                        image_info['image_uri'] = image_uri
                        image_info['tag'] = tag
                        result_images.append(image_info)
                        self.logger.info(f"📦 Added image: {tag}")
            
            self.logger.info(f"✅ Found {len(result_images)} valid images from {repo_name}")
            return result_images
        except Exception as e:
            self.logger.error(f"❌ Failed to get images from {repo_name}: {e}")
            return []
        
    def get_latest_training_gpu_image(self) -> Optional[Dict]:
        """Get the most recent training GPU image from beta-autogluon-training repository"""
        self.logger.info("🔍 Getting the most recent training GPU image from beta repository...")
        images = self.get_latest_autogluon_images('beta-autogluon-training', 1, None, gpu_only=True)
        if images:
            self.logger.info(f"📦 Found most recent GPU training image: {images[0]['tag']}")
            return images[0]
        self.logger.warning("⚠️ No GPU training images found in beta repository")
        return None
    
    def convert_python_version(self, py_version: str) -> str:
        """Convert py311 to 3.11 format"""
        if py_version.startswith('py'):
            version_num = py_version[2:]
            if len(version_num) == 3:
                return f"{version_num[0]}.{version_num[1:]}"
            elif len(version_num) == 2:
                return f"{version_num[0]}.{version_num[1]}"
        return py_version
    
    def update_available_images_md(self, training_images: Dict[str, Dict], inference_images: Dict[str, Dict]) -> bool:
        """Update available_images.md with new AutoGluon version information"""
        try:
            self.logger.info("📝 Updating available_images.md...")
            if not self.available_images_file.exists():
                self.logger.error(f"❌ File not found: {self.available_images_file}")
                return False
            with open(self.available_images_file, 'r') as f:
                content = f.read()
            self.logger.info(f"📋 Original file size: {len(content)} characters")
            if not isinstance(training_images, dict) or not isinstance(inference_images, dict):
                self.logger.error(f"❌ Expected dictionaries but got: training_images={type(training_images)}, inference_images={type(inference_images)}")
                return False
            if 'gpu' not in training_images or 'gpu' not in inference_images:
                return False
            if self.is_major_release:
                self.logger.info("🔄 Major version update detected - moving current to Prior sections")
                content = self.move_current_to_prior(content)
            content = self.update_main_sections_with_actual_images(content, training_images, inference_images)
            self.logger.info(f"📋 Updated file size: {len(content)} characters")
            with open(self.available_images_file, 'w') as f:
                f.write(content)
            self.logger.info("✅ available_images.md updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update available_images.md: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def move_current_to_prior(self, content: str) -> str:
        """Move current AutoGluon sections to Prior sections for major version updates"""
        lines = content.split('\n')
        new_lines = []
        training_section = self.extract_section(lines, "AutoGluon Training Containers")
        inference_section = self.extract_section(lines, "AutoGluon Inference Containers")
        i = 0
        while i < len(lines):
            line = lines[i]
            if "AutoGluon Training Containers" in line:
                new_lines.append(line)
                i += 1
                while i < len(lines) and not lines[i].startswith("AutoGluon Inference Containers"):
                    if not lines[i].startswith("AutoGluon Training Containers"):
                        new_lines.append(lines[i])
                    i += 1
                continue
            elif "AutoGluon Inference Containers" in line:
                new_lines.append(line)
                i += 1
                while i < len(lines) and not (lines[i].startswith("Prior AutoGluon") or lines[i].startswith("(--------")):
                    if not lines[i].startswith("AutoGluon Inference Containers"):
                        new_lines.append(lines[i])
                    i += 1
                continue
            elif "Prior AutoGluon Training Containers" in line:
                if training_section:
                    new_lines.extend(training_section)
                new_lines.append(line)
                i += 1
                continue
            elif "Prior AutoGluon Inference Containers" in line:
                if inference_section:
                    new_lines.extend(inference_section)
                new_lines.append(line)
                i += 1
                continue
            else:
                new_lines.append(line)
                i += 1
        return '\n'.join(new_lines)
    
    def extract_section(self, lines: list[str], section_name: str) -> list[str]:
        """Extract a table section from the markdown"""
        section_lines = []
        i = 0
        found_section = False
        while i < len(lines):
            line = lines[i]
            if section_name in line:
                found_section = True
                section_lines.append(line)
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if (next_line.startswith("AutoGluon") and "Containers" in next_line and 
                        section_name not in next_line):
                        break
                    if next_line.startswith("Prior AutoGluon") or next_line.startswith("(--------"):
                        break
                    section_lines.append(next_line)
                    i += 1
                break
            i += 1
        return section_lines if found_section else []
    
    def update_main_sections_with_actual_images(self, content: str, training_images: Dict[str, Dict], inference_images: Dict[str, Dict]) -> str:
        """Update the main AutoGluon sections with actual image URIs"""
        lines = content.split('\n')
        new_lines = []
        self.logger.info("🔍 Updating main sections with actual image URIs")
        autogluon_lines = []
        for i, line in enumerate(lines):
            if "AutoGluon" in line and "Containers" in line:
                autogluon_lines.append(f"Line {i}: {line.strip()}")
        if autogluon_lines:
            self.logger.info(f"📋 Found {len(autogluon_lines)} AutoGluon container sections:")
            for line in autogluon_lines:
                self.logger.info(f"   {line}")
        training_section_found = False
        inference_section_found = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == "AutoGluon Training Containers":
                self.logger.info(f"📝 Found AutoGluon Training Containers section at line {i}")
                training_section_found = True
                new_lines.append(line)
                new_lines.append("===============================")
                new_lines.append("")  
                new_lines.append("| Framework       | AutoGluon Version  | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |")
                new_lines.append("|-----------------|--------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|")
                if 'gpu' in training_images:
                    gpu_row = self.create_table_row_with_actual_uri(training_images['gpu'], "training", "GPU")
                    new_lines.append(gpu_row)
                if 'cpu' in training_images:
                    cpu_row = self.create_table_row_with_actual_uri(training_images['cpu'], "training", "CPU")
                    new_lines.append(cpu_row)
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if (next_line and 
                        not next_line.startswith("|") and 
                        not next_line.startswith("=") and
                        not next_line.startswith("-") and
                        next_line != ""):
                        break
                    i += 1
                continue
            elif line.strip() == "AutoGluon Inference Containers":
                self.logger.info(f"📝 Found AutoGluon Inference Containers section at line {i}")
                inference_section_found = True
                new_lines.append(line)
                new_lines.append("===============================")
                new_lines.append("") 
                new_lines.append("| Framework       | AutoGluon Version  | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |")
                new_lines.append("|-----------------|--------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|")
                if 'gpu' in inference_images:
                    gpu_row = self.create_table_row_with_actual_uri(inference_images['gpu'], "inference", "GPU")
                    new_lines.append(gpu_row)
                if 'cpu' in inference_images:
                    cpu_row = self.create_table_row_with_actual_uri(inference_images['cpu'], "inference", "CPU")
                    new_lines.append(cpu_row)
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if (next_line and 
                        not next_line.startswith("|") and 
                        not next_line.startswith("=") and
                        not next_line.startswith("-") and
                        next_line != ""):
                        break
                    i += 1
                continue
                
            else:
                new_lines.append(line)
                i += 1
        
        if not training_section_found:
            self.logger.warning("⚠️ AutoGluon Training Containers section not found!")
            self.logger.warning("Lines containing 'AutoGluon Training':")
            for i, line in enumerate(lines):
                if "AutoGluon Training" in line:
                    self.logger.warning(f"  Line {i}: '{line.strip()}'")
                    
        if not inference_section_found:
            self.logger.warning("⚠️ AutoGluon Inference Containers section not found!")
            self.logger.warning("Lines containing 'AutoGluon Inference':")
            for i, line in enumerate(lines):
                if "AutoGluon Inference" in line:
                    self.logger.warning(f"  Line {i}: '{line.strip()}'")
            
        self.logger.info(f"✅ Main sections updated. Lines: {len(lines)} -> {len(new_lines)}")
        return '\n'.join(new_lines)
    
    def create_table_row_with_actual_uri(self, image_info: Dict, job_type: str, compute_type: str) -> str:
        """Create a table row using the actual image URI extracted from repository"""
        python_version = self.convert_python_version(image_info['python_versions'][0])
        actual_uri = image_info['image_uri']
        
        self.logger.info(f"📋 Creating {job_type} {compute_type} row with actual URI: {actual_uri}")
        
        return f"| AutoGluon {self.current_version} | {self.current_version}              | {job_type} | {compute_type}     | {python_version} ({image_info['python_versions'][0]})             | {actual_uri} |"
    
    def move_current_to_prior(self, content: str) -> str:
        """Move current AutoGluon sections to Prior sections for major version updates"""
        lines = content.split('\n')
        new_lines = []
        
        self.logger.info("🔄 Moving current AutoGluon sections to Prior sections")
        
        # Extract current training section
        current_training_section = self.extract_current_section(lines, "AutoGluon Training Containers")
        current_inference_section = self.extract_current_section(lines, "AutoGluon Inference Containers")
        
        self.logger.info(f"📋 Extracted training section: {len(current_training_section)} lines")
        self.logger.info(f"📋 Extracted inference section: {len(current_inference_section)} lines")
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Prior AutoGluon Training Containers" in line:
                self.logger.info("📝 Found Prior AutoGluon Training Containers section")
                if current_training_section:
                    new_lines.extend(current_training_section)
                    new_lines.append("") 
                new_lines.append(line)
                i += 1
                continue
            elif "Prior AutoGluon Inference Containers" in line:
                self.logger.info("📝 Found Prior AutoGluon Inference Containers section")
                if current_inference_section:
                    new_lines.extend(current_inference_section)
                    new_lines.append("") 
                new_lines.append(line)
                i += 1
                continue
            elif i == len(lines) - 1:
                new_lines.append(line)
                if current_training_section:
                    new_lines.append("")
                    new_lines.extend(current_training_section)
                if current_inference_section:
                    new_lines.append("")
                    new_lines.extend(current_inference_section)
                i += 1
                continue
                
            else:
                new_lines.append(line)
                i += 1
        
        self.logger.info("✅ Current sections moved to Prior sections")
        return '\n'.join(new_lines)
    
    def extract_current_section(self, lines: list[str], section_name: str) -> list[str]:
        """Extract a current AutoGluon section from the markdown"""
        section_lines = []
        i = 0
        found_section = False
        
        while i < len(lines):
            line = lines[i]
            if section_name in line and "Prior" not in line:
                found_section = True
                prior_header = section_name.replace("AutoGluon", "Prior AutoGluon")
                section_lines.append(prior_header)
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if ((next_line.startswith("AutoGluon") and "Containers" in next_line and section_name not in next_line) or
                        next_line.startswith("Prior AutoGluon") or 
                        next_line.startswith("(--------") or
                        next_line.startswith("---")):
                        break
                    section_lines.append(next_line)
                    i += 1
                break
            i += 1
        
        if found_section:
            self.logger.info(f"✅ Extracted {section_name} section with {len(section_lines)} lines")
        else:
            self.logger.warning(f"⚠️ Could not find {section_name} section")
        
        return section_lines if found_section else []
            
    def parse_image_tag(self, tag: str, require_cuda: bool = True) -> Optional[Dict]:
        """Parse image tag to extract os_version, python_versions, and cuda_version"""
        try:
            self.logger.info(f"🔍 Parsing image tag: {tag}")
            os_match = re.search(r'ubuntu(\d+)\.(\d+)', tag)
            if os_match:
                os_version = f"ubuntu{os_match.group(1)}.{os_match.group(2)}"
                self.logger.info(f"   Found OS version: {os_version}")
            else:
                self.logger.error(f"❌ Could not extract OS version from tag: {tag}")
                return None
            python_match = re.search(r'-py(\d+)-', tag)
            if python_match:
                python_version = f"py{python_match.group(1)}"
                self.logger.info(f"   Found Python version: {python_version}")
            else:
                self.logger.error(f"❌ Could not extract Python version from tag: {tag}")
                return None
            cuda_match = re.search(r'cu(\d+)', tag)
            if cuda_match:
                cuda_version = f"cu{cuda_match.group(1)}"
                self.logger.info(f"   Found CUDA version: {cuda_version}")
            elif require_cuda:
                self.logger.warning(f"⚠️ No CUDA version found in tag: {tag} (skipping - likely CPU image)")
                return None
            else:
                cuda_version = ""
                self.logger.info(f"   No CUDA version (CPU image)")
            return {
                'os_version': os_version,
                'python_versions': [python_version],
                'cuda_version': cuda_version
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to parse image tag {tag}: {e}")
            return None
            
    def get_next_release_number(self, yaml_file: Path) -> int:
        """Get the next release number from the YAML file"""
        try:
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            if 'release_images' not in content:
                return 1
            max_num = 0
            for key in content['release_images'].keys():
                if isinstance(key, int):
                    max_num = max(max_num, key)
            return max_num + 1
        except Exception as e:
            self.logger.error(f"❌ Failed to get next release number from {yaml_file}: {e}")
            return 18 
            
    def backup_yaml_files(self):
        """Backup original YAML file contents"""
        try:
            with open(self.training_file, 'r') as f:
                self.original_training_content = f.read()
            with open(self.inference_file, 'r') as f:
                self.original_inference_content = f.read()
            self.logger.info("✅ Original YAML files backed up")
        except Exception as e:
            self.logger.error(f"❌ Failed to backup YAML files: {e}")
            raise
            
    def backup_available_images_file(self):
        """Backup original available_images.md file content"""
        try:
            with open(self.available_images_file, 'r') as f:
                self.original_available_images_content = f.read()
            self.logger.info("✅ Original available_images.md file backed up")
        except Exception as e:
            self.logger.error(f"❌ Failed to backup available_images.md: {e}")
            raise
            
    def revert_yaml_files(self):
        """Revert YAML files to original content"""
        try:
            if self.original_training_content:
                with open(self.training_file, 'w') as f:
                    f.write(self.original_training_content)
            if self.original_inference_content:
                with open(self.inference_file, 'w') as f:
                    f.write(self.original_inference_content)
            self.logger.info("✅ YAML files reverted to original state")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to revert YAML files: {e}")
            return False
            
    def commit_and_push_yaml_changes(self, commit_message: str):
        """Commit and push YAML file changes to the branch"""
        try:
            original_dir = os.getcwd()
            os.chdir(self.repo_dir)
            subprocess.run(['git', 'add', 'release_images_training.yml', 'release_images_inference.yml'], check=True)
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            branch_name = f"autogluon-{self.current_version}-release"
            subprocess.run(['git', 'push', 'origin', branch_name], check=True)
            self.logger.info(f"✅ YAML changes committed and pushed: {commit_message}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to commit and push YAML changes: {e}")
            return False
        finally:
            os.chdir(original_dir)
            
    def commit_and_push_available_images_changes(self, commit_message: str):
        """Commit and push available_images.md changes to the branch"""
        try:
            original_dir = os.getcwd()
            os.chdir(self.repo_dir)
            
            # Add the modified available_images.md file
            subprocess.run(['git', 'add', 'available_images.md'], check=True)
            
            # Commit changes
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Push changes
            branch_name = f"autogluon-{self.current_version}-release"
            subprocess.run(['git', 'push', 'origin', branch_name], check=True)
            self.logger.info(f"✅ available_images.md changes committed and pushed: {commit_message}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to commit and push available_images.md changes: {e}")
            return False
        finally:
            os.chdir(original_dir)
            
    def update_yaml_file(self, yaml_file: Path, image_info: Dict, file_type: str):
        """Update a single YAML file with AutoGluon configuration"""
        try:
            with open(yaml_file, 'r') as f:
                original_content = f.read()
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            if 'release_images' not in content:
                content['release_images'] = {}
            next_num = self.get_next_release_number(yaml_file)
            autogluon_section = f"""  {next_num}:
    framework: "autogluon"
    version: "{self.current_version}"
    arch_type: "x86"
    {file_type}:
      device_types: ["cpu", "gpu"]
      python_versions: ["{image_info['python_versions'][0]}"]
      os_version: "{image_info['os_version']}"
      cuda_version: "{image_info['cuda_version']}"
      example: False
      disable_sm_tag: False
      force_release: False"""
            if not original_content.startswith('---'):
                new_content = f"---\n{original_content.rstrip()}\n{autogluon_section}\n"
            else:
                new_content = f"{original_content.rstrip()}\n{autogluon_section}\n"
            with open(yaml_file, 'w') as f:
                f.write(new_content)
            self.logger.info(f"✅ Updated {yaml_file.name} with AutoGluon config (release #{next_num})")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update {yaml_file}: {e}")
            return False
            
    def update_release_images_files(self, image_info: Dict):
        """Update both training and inference YAML files"""
        success = True
        if not self.update_yaml_file(self.training_file, image_info, 'training'):
            success = False
        if not self.update_yaml_file(self.inference_file, image_info, 'inference'):
            success = False
        return success
        
    def wait_for_pr_merge(self, pr_type: str):
        """Wait for PR to be merged"""
        print(f"📋 Waiting for {pr_type} PR to be merged...")
        while True:
            if self.prompt_user(f"Is the {pr_type} PR merged?"):
                print(f"✅ {pr_type} PR confirmed as merged. Proceeding...")
                break
            else:
                print(f"⏳ Waiting for {pr_type} PR merge...")
                
    def run_release_images_automation(self):
        """Main automation workflow"""
        try:
            print("🚀 Starting AutoGluon Release Images Automation...")
            self.wait_for_cr_completion()
            print("🔍 Extracting image information for YAML files...")
            print(f"📋 Using environment ACCOUNT_ID for beta repositories")
            image_info = self.get_latest_training_gpu_image()
            if not image_info:
                raise Exception("Failed to get training image information")
            print(f"📦 Image info extracted from {image_info['tag']}:")
            print(f"   OS Version: {image_info['os_version']}")
            print(f"   Python Versions: {image_info['python_versions']}")
            print(f"   CUDA Version: {image_info['cuda_version']}")
            self.backup_yaml_files()
            print("📝 Updating release_images files...")
            if not self.update_release_images_files(image_info):
                raise Exception("Failed to update release_images files")
                
            print("✅ Release images files updated successfully")
            if not self.prompt_user("Commit and create PR with YAML changes?"):
                print("❌ Operation cancelled by user")
                return False
            commit_message = f"AutoGluon {self.current_version}: Add release images configuration"
            if not self.commit_and_push_yaml_changes(commit_message):
                raise Exception("Failed to commit and push YAML changes")
            print("🚀 Creating Pull Request for YAML changes...")
            if not self.pr_automation.create_pull_request():
                raise Exception("Failed to create Pull Request")
            self.wait_for_pr_merge("YAML")
            print("🔄 Reverting YAML changes...")
            if not self.revert_yaml_files():
                raise Exception("Failed to revert YAML files")
            revert_commit_message = f"AutoGluon {self.current_version}: Revert release images configuration"
            if not self.commit_and_push_yaml_changes(revert_commit_message):
                raise Exception("Failed to commit and push YAML revert")
            print("🚀 Creating Revert Pull Request for YAML changes...")
            if not self.pr_automation.create_pull_request():
                raise Exception("Failed to create Revert Pull Request")
            self.wait_for_pr_merge("YAML revert")
            print("📝 Starting available_images.md update process...")
            print("🔍 Extracting actual image tags from production repositories...")
            print(f"📋 Using production account ID: {self.PRODUCTION_ACCOUNT_ID}")
            training_images = self.get_latest_autogluon_images_by_type('autogluon-training', self.PRODUCTION_ACCOUNT_ID)
            inference_images = self.get_latest_autogluon_images_by_type('autogluon-inference', self.PRODUCTION_ACCOUNT_ID)
            if not training_images or not inference_images:
                raise Exception("Failed to get sufficient images from production repositories")
            print(f"📦 Found training images: {list(training_images.keys()) if isinstance(training_images, dict) else type(training_images)}")
            print(f"📦 Found inference images: {list(inference_images.keys()) if isinstance(inference_images, dict) else type(inference_images)}")
            for img_type, img_data in training_images.items():
                self.logger.info(f"📦 Training {img_type} image URI: {img_data.get('image_uri', 'N/A')}")
            for img_type, img_data in inference_images.items():
                self.logger.info(f"📦 Inference {img_type} image URI: {img_data.get('image_uri', 'N/A')}")
            self.backup_available_images_file()
            print("📝 Updating available_images.md with actual image URIs from repositories...")
            if not self.update_available_images_md(training_images, inference_images):
                raise Exception("Failed to update available_images.md")
            print("✅ available_images.md updated successfully with actual image URIs")
            if not self.prompt_user("Commit and create PR with available_images.md changes?"):
                print("❌ Operation cancelled by user")
                return False
            available_images_commit_message = f"AutoGluon {self.current_version}: Update available images documentation with actual image URIs"
            if not self.commit_and_push_available_images_changes(available_images_commit_message):
                raise Exception("Failed to commit and push available_images.md changes")
            print("🚀 Creating Pull Request for available_images.md...")
            if not self.pr_automation.create_pull_request():
                raise Exception("Failed to create available_images.md Pull Request")                
            self.wait_for_pr_merge("available_images.md")
            print("✅ AutoGluon Release Images Automation completed successfully!")
            print("📋 Summary:")
            print("   - YAML files: Updated and reverted (temporary)")
            print("   - available_images.md: Updated permanently with actual image URIs from repositories")
            print("   - Total PRs created: 3")
            return True
        except Exception as e:
            self.logger.error(f"❌ Automation failed: {e}")
            print(f"❌ Error: {e}")
            return False

def main():
    """Main function for AutoGluon Release Images Automation"""
    import argparse
    parser = argparse.ArgumentParser(description='AutoGluon Release Images Automation - Updates YAML files and available_images.md')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL') 
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print("🚀 AutoGluon Release Images Automation")
    automation = AutoGluonReleaseImagesAutomation(
        args.current_version,
        args.previous_version,
        args.fork_url
    )
    success = automation.run_release_images_automation()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()