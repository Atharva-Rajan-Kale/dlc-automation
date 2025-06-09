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
    pytorch_version: str=""

class AutoGluonReleaseAutomation:
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        self.current_version = current_version
        self.previous_version = previous_version
        self.fork_url = fork_url
        current_parts = current_version.split('.')
        previous_parts = previous_version.split('.')
        self.is_major_release = (current_parts[0] != previous_parts[0] or 
                               (current_parts[1] != previous_parts[1] and current_parts[2] == '0'))
        self.short_version = current_version.replace('.', '')
        original_dir = Path(os.getcwd())
        self.workspace_dir = original_dir / "deep-learning-container"
        self.repo_dir = self.workspace_dir / "deep-learning-containers"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Workspace directory: {self.workspace_dir}")
        self.logger.info(f"Repository directory: {self.repo_dir}")

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
                            self._parse_image_metadata(ecr_image)
                            images.append(ecr_image)
                images.sort(key=lambda x: x.pushed_at, reverse=True)
                return images
            except Exception as e:
                print(f"❌ Error getting {repository} images: {e}")
                print("💡 Make sure your AWS credentials have ECR read permissions")
                return []
        
        def _parse_image_metadata(self, image: ECRImage):
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

    def step1_create_branch(self):
        """Step 1: Cut a new branch in fork to work on a new release"""
        self.logger.info("Step 1: Creating release branch")
        self.workspace_dir.mkdir(exist_ok=True)
        original_dir = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            if not Path("deep-learning-containers").exists():
                self.logger.info(f"Cloning from {self.fork_url}")
                subprocess.run(["git", "clone", self.fork_url, "deep-learning-containers"], check=True)
            os.chdir("deep-learning-containers")
            try:
                subprocess.run(["git", "remote", "get-url", "upstream"], capture_output=True, check=True)
                self.logger.info("Upstream remote already exists")
            except:
                self.logger.info("Adding upstream remote")
                subprocess.run(["git", "remote", "add", "upstream", 
                              "https://github.com/aws/deep-learning-containers.git"], check=True)
            self.logger.info("Syncing with upstream...")
            subprocess.run(["git", "fetch", "upstream"], check=True)
            subprocess.run(["git", "checkout", "master"], check=True)
            subprocess.run(["git", "reset", "--hard", "upstream/master"], check=True)
            branch_name = f"autogluon-{self.current_version}-release"
            self.logger.info(f"Creating branch: {branch_name}")
            try:
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            except:
                subprocess.run(["git", "checkout", branch_name], check=True)
            
            self.logger.info("✅ Step 1 completed: Release branch created")
        except Exception as e:
            os.chdir(original_dir)
            raise e

    def step2_update_toml(self):
        """Step 2: Update toml file to build only AutoGluon"""
        self.logger.info("Step 2: Updating TOML configuration")
        toml_path = Path("dlc_developer_config.toml")
        if not toml_path.exists():
            self.logger.error(f"TOML file not found: {toml_path.absolute()}")
            return
        with open(toml_path, 'r') as f:
            content = f.read()
        content = re.sub(
            r'build_frameworks\s*=\s*\[.*?\]',
            'build_frameworks = ["autogluon"]',
            content,
            flags=re.DOTALL
        )
        self.logger.info("Updated build_frameworks = ['autogluon']")
        content = re.sub(
            r'dlc-pr-autogluon-training\s*=\s*""',
            'dlc-pr-autogluon-training = "autogluon/training/buildspec.yml"',
            content
        )
        self.logger.info("Updated dlc-pr-autogluon-training buildspec path")
        content = re.sub(
            r'dlc-pr-autogluon-inference\s*=\s*""',
            'dlc-pr-autogluon-inference = "autogluon/inference/buildspec.yml"',
            content
        )
        self.logger.info("Updated dlc-pr-autogluon-inference buildspec path")
        with open(toml_path, 'w') as f:
            f.write(content)
        subprocess.run(["git", "add", str(toml_path)], check=True)
        subprocess.run(["git", "commit", "-m", 
                       f"AutoGluon {self.current_version}: Update TOML for AutoGluon-only build"], 
                      check=True)
        self.logger.info("✅ Step 2 completed: TOML updated and committed")
    
    def step3_create_docker_resources(self):
        """Step 3: Create new release docker resources"""
        self.logger.info("Step 3: Creating docker resources")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        current_files = list(Path(".").iterdir())
        self.logger.info(f"Files in current directory: {[f.name for f in current_files]}")
        if not Path("autogluon").exists():
            self.logger.error("autogluon directory not found in current directory")
            if Path("deep-learning-containers/autogluon").exists():
                self.logger.info("Found autogluon in deep-learning-containers subdirectory, changing to it")
                os.chdir("deep-learning-containers")
                self.logger.info(f"Now in directory: {os.getcwd()}")
            else:
                self.logger.error("autogluon directory not found anywhere")
                return False
        self.logger.info("Selecting optimal base images from ECR...")
        ecr_selector = self.ECRImageSelector()
        image_selection = ecr_selector.select_matching_cuda_images()
        if not image_selection:
            self.logger.error("Could not find matching CUDA images")
            return False
        cuda_version = image_selection['cuda_version']
        self.logger.info(f"Selected CUDA version: {cuda_version}")
        self.logger.info(f"Training CPU: {image_selection['training_cpu'].tag}")
        self.logger.info(f"Training GPU: {image_selection['training_gpu'].tag}")
        self.logger.info(f"Inference CPU: {image_selection['inference_cpu'].tag}")
        self.logger.info(f"Inference GPU: {image_selection['inference_gpu'].tag}")
        self.selected_images = image_selection
        if self.is_major_release:
            success = self._create_major_docker_resources(image_selection)
        else:
            success = self._update_minor_docker_resources(image_selection)
        if success:
            self.logger.info("✅ Step 3 completed: Docker resources created (not committed)")
        else:
            self.logger.error("❌ Step 3 failed: Could not create docker resources")
        return success
    
    def step4_update_buildspec_files(self):
        """Step 4: Update buildspec.yml files"""
        self.logger.info("Step 4: Updating buildspec files")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        if not hasattr(self, 'selected_images') or not self.selected_images:
            self.logger.error("No selected images from step 3. Run step 3 first.")
            return False
        image_info = self._extract_buildspec_info_from_images()
        self.logger.info(f"Extracted image info: {image_info}")
        training_success = self._update_buildspec("training", image_info)
        inference_success = self._update_buildspec("inference", image_info)
        if training_success and inference_success:
            self.logger.info("✅ Step 4 completed: Buildspec files updated (not committed)")
            return True
        else:
            self.logger.error("❌ Step 4 failed: Could not update buildspec files")
            return False
    
    def step5_package_model(self):
        """Step 5: Update package_model.py version and execute it"""
        self.logger.info("Step 5: Packaging model")
        main_project_dir = self.repo_dir.parent.parent
        package_model_path = main_project_dir / "package_model.py"
        self.logger.info(f"Repo dir: {self.repo_dir}")
        self.logger.info(f"Looking for package_model.py in: {main_project_dir}")
        if not package_model_path.exists():
            self.logger.error(f"package_model.py not found at: {package_model_path}")
            return False
        try:
            with open(package_model_path, 'r') as f:
                content = f.read()
            content = re.sub(r"version\s*=\s*['\"][\d.]+['\"]", f"version = '{self.current_version}'", content)
            with open(package_model_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Updated version to {self.current_version}")
            original_cwd = os.getcwd()
            os.chdir(main_project_dir)
            self.logger.info(f"Changed to: {os.getcwd()}")
            self.logger.info("Executing package_model.py...")
            result = subprocess.run(['python', 'package_model.py'], check=True)
            self.logger.info("✅ Model training completed")
            source_file = main_project_dir / f"model_{self.current_version}.tar.gz"
            target_dir = self.repo_dir / "test/sagemaker_tests/autogluon/inference/resources/model"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"model_{self.current_version}.tar.gz"
            shutil.move(str(source_file), str(target_file))
            self.logger.info(f"✅ Moved model to: {target_file}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Step 5 failed: {e}")
            return False
        finally:
            os.chdir(original_cwd)
            self.logger.info(f"Returned to: {os.getcwd()}")
    
    def step6_build_upload_docker(self):
        """Step 6: Fix code issues, build and upload Docker images to ECR"""
        self.logger.info("Step 6: Building and uploading Docker images")
        
        # Check if ACCOUNT_ID and REGION are set in environment
        account_id = os.environ.get('ACCOUNT_ID')
        region = os.environ.get('REGION')
        
        if not account_id:
            self.logger.error("❌ ACCOUNT_ID environment variable not set")
            self.logger.info("💡 Please run: export ACCOUNT_ID=your_account_id")
            return False
            
        if not region:
            self.logger.error("❌ REGION environment variable not set")
            self.logger.info("💡 Please run: export REGION=us-east-1")
            return False
        
        self.logger.info(f"Using ACCOUNT_ID: {account_id}")
        self.logger.info(f"Using REGION: {region}")
        original_dir = os.getcwd()
        
        try:
            if not self.repo_dir.exists():
                self.logger.error(f"Repository directory not found :{self.repo_dir}")
                return False
            os.chdir(self.repo_dir)
            self.logger.info(f"Changed to repo directory: {os.getcwd()}")
            # Fix code issues
            if not self._fix_code_issues():
                return False
            
            # Set up build environment
            if not self._setup_build_environment(account_id, region):
                return False
            
            # Build and upload Docker images
            if not self._build_docker_images():
                return False
            
            self.logger.info("✅ Step 6 completed: Docker images built and uploaded")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step 6 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
            self.logger.info(f"Returned to original directory :{os.getcwd()}")

    def _fix_code_issues(self):
        """Fix known issues in the codebase"""
        self.logger.info("🔧 Fixing code issues...")
        
        try:
            # Fix 1: patch_helper.py import issue
            patch_helper_path = self.repo_dir/"src/patch_helper.py"
            if patch_helper_path.exists():
                with open(patch_helper_path, 'r') as f:
                    content = f.read()
                
                if "from src.constants import PATCHING_INFO_PATH_WITHIN_DLC" in content:
                    content = content.replace(
                        "from src.constants import PATCHING_INFO_PATH_WITHIN_DLC",
                        "from constants import PATCHING_INFO_PATH_WITHIN_DLC"
                    )
                    
                    with open(patch_helper_path, 'w') as f:
                        f.write(content)
                    
                    self.logger.info("✅ Fixed patch_helper.py import")
                else:
                    self.logger.info("ℹ️  patch_helper.py import already fixed")
            else:
                self.logger.warning("⚠️  patch_helper.py not found")
            
            # Fix 2: utils.py path issue
            utils_path = self.repo_dir/"src/utils.py "
            if utils_path.exists():
                with open(utils_path, 'r') as f:
                    content = f.read()
                
                # Check if the fix is already there
                if "_project_root" not in content:
                    # Find "import sys" line and add our code after it
                    lines = content.split('\n')
                    sys_import_index = -1
                    
                    for i, line in enumerate(lines):
                        if line.strip() == "import sys":
                            sys_import_index = i
                            break
                    
                    if sys_import_index != -1:
                        # Insert the new lines after import sys
                        new_lines = [
                            "project_root = os.path.dirname(os.path.dirname(os.path.abspath(file_)))",
                            "if _project_root not in sys.path:",
                            "    sys.path.insert(0, _project_root)"
                        ]
                        
                        # Insert at the correct position
                        for j, new_line in enumerate(new_lines):
                            lines.insert(sys_import_index + 1 + j, new_line)
                        
                        # Write back
                        with open(utils_path, 'w') as f:
                            f.write('\n'.join(lines))
                        
                        self.logger.info("✅ Fixed utils.py path issue")
                    else:
                        self.logger.warning("⚠️  Could not find 'import sys' in utils.py")
                else:
                    self.logger.info("ℹ️  utils.py path issue already fixed")
            else:
                self.logger.warning("⚠️  utils.py not found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fix code issues: {e}")
            return False

    def _setup_build_environment(self, account_id: str, region: str):
        """Set up environment variables and dependencies"""
        self.logger.info("🛠️  Setting up build environment...")
        
        try:
            # Environment variables are already set by user, just verify
            self.logger.info(f"Using ACCOUNT_ID={account_id}, REGION={region}")
            
            # ECR login to user's account
            self.logger.info("🔐 Logging into your ECR...")
            login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
            subprocess.run(login_cmd, shell=True, check=True)
            self.logger.info("✅ Logged into your ECR")
            
            # ECR login to AWS's official account
            self.logger.info("🔐 Logging into AWS's ECR...")
            aws_login_cmd = f"aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com"
            subprocess.run(aws_login_cmd, shell=True, check=True)
            self.logger.info("✅ Logged into AWS's ECR")
            
            # Create and activate virtual environment
            self.logger.info("🐍 Setting up Python environment...")
            if Path("dlc").exists():
                shutil.rmtree("dlc")
            
            subprocess.run(["python3", "-m", "venv", "dlc"], check=True)
            
            # Install dependencies
            subprocess.run(["dlc/bin/pip", "install", "-U", "pip"], check=True)
            subprocess.run(["dlc/bin/pip", "install", "-U", "setuptools", "wheel"], check=True)
            subprocess.run(["dlc/bin/pip", "install", "-r", "src/requirements.txt"], check=True)
            self.logger.info("✅ Python dependencies installed")

            subprocess.run(["dlc/bin/pip","install","requests==2.31.0"],check=True)
            self.logger.info("Python dependencies and requests version installed")
            
            # Run setup script
            self.logger.info("🔧 Running setup script...")
            subprocess.run(["bash", "src/setup.sh", "autogluon"], check=True)
            self.logger.info("✅ Setup script completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup build environment: {e}")
            return False

    def _build_docker_images(self):
        """Build and upload all Docker images"""
        self.logger.info("🏗️  Building Docker images...")
        
        try:
            # Build training images
            os.environ['REPOSITORY_NAME'] = 'beta-autogluon-training'
            self.logger.info("Building training CPU image...")
            
            subprocess.run([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/training/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "training",
                "--device_types", "cpu",
                "--py_versions", "py3"
            ], check=True)
            self.logger.info("✅ Training CPU image built")
            
            self.logger.info("Building training GPU image...")
            subprocess.run([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/training/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "training",
                "--device_types", "gpu",
                "--py_versions", "py3"
            ], check=True)
            self.logger.info("✅ Training GPU image built")
            
            # Build inference images
            os.environ['REPOSITORY_NAME'] = 'beta-autogluon-inference'
            self.logger.info("Building inference CPU image...")
            
            subprocess.run([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/inference/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "inference",
                "--device_types", "cpu",
                "--py_versions", "py3"
            ], check=True)
            self.logger.info("✅ Inference CPU image built")
            
            self.logger.info("Building inference GPU image...")
            subprocess.run([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/inference/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "inference",
                "--device_types", "gpu",
                "--py_versions", "py3"
            ], check=True)
            self.logger.info("✅ Inference GPU image built")
            
            self.logger.info("🎉 All Docker images built and uploaded successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build Docker images: {e}")
            return False

    def _extract_buildspec_info_from_images(self):
        """Extract version info from selected images for buildspec updates"""
        sample_image = self.selected_images['training_cpu']
        self.logger.info(f"Extracting info from sample image tag: {sample_image.tag}")
        python_match = re.search(r'-py(\d+)-', sample_image.tag)
        python_version = f"py{python_match.group(1)}" if python_match else "py311"
        os_match = re.search(r'ubuntu(\d+\.\d+)', sample_image.tag)
        os_version = f"ubuntu{os_match.group(1)}" if os_match else "ubuntu22.04"
        cuda_version = self.selected_images['cuda_version']
        pytorch_version = self.selected_images['pytorch_version']
        return {
            'python_version': python_version,
            'os_version': os_version, 
            'cuda_version': cuda_version,
            'pytorch_version': pytorch_version
        }

    def _update_buildspec(self, container_type, image_info):
        """Update buildspec.yml for training or inference"""
        buildspec_path = Path(f"autogluon/{container_type}/buildspec.yml")
        if not buildspec_path.exists():
            self.logger.error(f"Buildspec not found: {buildspec_path}")
            return False
        self.logger.info(f"Updating buildspec: {buildspec_path}")
        with open(buildspec_path, 'r') as f:
            content = f.read()
        original_content = content
        if self.is_major_release:
            backup_name = f"buildspec-{self.previous_version.replace('.', '-')}.yml"
            backup_path = buildspec_path.parent / backup_name
            with open(backup_path, 'w') as f:
                f.write(content)
            self.logger.info(f"📁 Created backup: {backup_path}")
            curr_short = '.'.join(self.current_version.split('.')[:2])  
            content = re.sub(
                r'(short_version:\s*&SHORT_VERSION\s+)[\d.]+',
                rf'\g<1>{curr_short}',
                content
            )
            self.logger.info(f"Updated short_version to: {curr_short}")
        content = re.sub(
            r'(version:\s*&VERSION\s+)[\d.]+',
            rf'\g<1>{self.current_version}',
            content
        )
        self.logger.info(f"Updated version to: {self.current_version}")
        content = re.sub(
            r'(tag_python_version:\s*&TAG_PYTHON_VERSION\s+)\w+',
            rf'\g<1>{image_info["python_version"]}',
            content
        )
        self.logger.info(f"Updated python version to: {image_info['python_version']}")
        content = re.sub(
            r'(cuda_version:\s*&CUDA_VERSION\s+)\w+',
            rf'\g<1>{image_info["cuda_version"]}',
            content
        )
        self.logger.info(f"Updated CUDA version to: {image_info['cuda_version']}")
        content = re.sub(
            r'(os_version:\s*&OS_VERSION\s+)[\w.]+',
            rf'\g<1>{image_info["os_version"]}',
            content
        )
        self.logger.info(f"Updated OS version to: {image_info['os_version']}")
        if content != original_content:
            with open(buildspec_path, 'w') as f:
                f.write(content)
            self.logger.info(f"✅ Successfully updated {buildspec_path}")
            self.logger.info(f"   📋 Changes made to {container_type} buildspec:")
            self.logger.info(f"   - Version: {self.current_version}")
            if self.is_major_release:
                curr_short = '.'.join(self.current_version.split('.')[:2])
                self.logger.info(f"   - Short version: {curr_short}")
            self.logger.info(f"   - Python: {image_info['python_version']}")
            self.logger.info(f"   - CUDA: {image_info['cuda_version']}")
            self.logger.info(f"   - OS: {image_info['os_version']}")
        else:
            self.logger.info(f"ℹ️  No changes needed for {buildspec_path}")
        return True
        
    def _create_major_docker_resources(self, image_selection: Dict) -> bool:
        """Create new directories for major version (e.g., 1.3 -> 1.4)"""
        self.logger.info("Creating resources for MAJOR version update")
        prev_major_minor = '.'.join(self.previous_version.split('.')[:2])  
        curr_major_minor = '.'.join(self.current_version.split('.')[:2])   
        source_training_dir = Path(f"autogluon/training/docker/{prev_major_minor}")
        source_inference_dir = Path(f"autogluon/inference/docker/{prev_major_minor}")
        target_training_dir = Path(f"autogluon/training/docker/{curr_major_minor}")
        target_inference_dir = Path(f"autogluon/inference/docker/{curr_major_minor}")
        if not source_training_dir.exists():
            self.logger.error(f"Source training directory does not exist: {source_training_dir}")
            return False
        if not source_inference_dir.exists():
            self.logger.error(f"Source inference directory does not exist: {source_inference_dir}")
            return False
        self.logger.info(f"Copying {source_training_dir} -> {target_training_dir}")
        self.logger.info(f"Copying {source_inference_dir} -> {target_inference_dir}")
        if target_training_dir.exists():
            shutil.rmtree(target_training_dir)
        if target_inference_dir.exists():
            shutil.rmtree(target_inference_dir)
        shutil.copytree(source_training_dir, target_training_dir)
        shutil.copytree(source_inference_dir, target_inference_dir)
        self.logger.info(f"✅ Copied directory structures for version {curr_major_minor}")
        cuda_num = image_selection['cuda_version'][2:]  
        self._update_dockerfiles_in_directory(target_training_dir, image_selection, "training", cuda_num)
        self._update_dockerfiles_in_directory(target_inference_dir, image_selection, "inference", cuda_num)
        self.logger.info(f"✅ Updated Dockerfiles for version {curr_major_minor}")
        return True

    def _update_minor_docker_resources(self, image_selection: Dict) -> bool:
        """Update existing directories for minor version (e.g., 1.3.0 -> 1.3.1)"""
        self.logger.info("Updating resources for MINOR version update")
        major_minor = '.'.join(self.current_version.split('.')[:2])  
        training_dir = Path(f"autogluon/training/docker/{major_minor}")
        inference_dir = Path(f"autogluon/inference/docker/{major_minor}")
        if not training_dir.exists():
            self.logger.error(f"Training directory not found: {training_dir}")
            return False
        if not inference_dir.exists():
            self.logger.error(f"Inference directory not found: {inference_dir}")
            return False
        
        self.logger.info(f"Updating existing directories: {training_dir} and {inference_dir}")
        cuda_num = image_selection['cuda_version'][2:]  
        self._update_dockerfiles_in_directory(training_dir, image_selection, "training", cuda_num)
        self._update_dockerfiles_in_directory(inference_dir, image_selection, "inference", cuda_num)
        self.logger.info(f"✅ Updated Dockerfiles in existing {major_minor} directories")
        return True

    def _update_dockerfiles_in_directory(self, base_dir: Path, image_selection: Dict, 
                                        container_type: str, cuda_num: str):
        """Update Dockerfiles in the specified directory structure"""
        gpu_image = image_selection[f'{container_type}_gpu']
        cpu_image = image_selection[f'{container_type}_cpu']
        self.logger.info(f"Updating Dockerfiles in {base_dir}")
        py3_dir = base_dir / "py3"
        if not py3_dir.exists():
            self.logger.error(f"py3 directory not found: {py3_dir}")
            return
        cpu_dockerfile = py3_dir / "Dockerfile.cpu"
        if cpu_dockerfile.exists():
            self._update_single_dockerfile(cpu_dockerfile, cpu_image.image_uri, "CPU")
        else:
            self.logger.warning(f"CPU Dockerfile not found: {cpu_dockerfile}")
        cuda_dir = py3_dir / f"cu{cuda_num}"
        if cuda_dir.exists():
            gpu_dockerfile = cuda_dir / "Dockerfile.gpu"
            if gpu_dockerfile.exists():
                self._update_single_dockerfile(gpu_dockerfile, gpu_image.image_uri, "GPU")
            else:
                self.logger.warning(f"GPU Dockerfile not found: {gpu_dockerfile}")
        else:
            existing_cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
            if existing_cuda_dirs:
                old_cuda_dir = existing_cuda_dirs[0]  
                self.logger.info(f"Renaming {old_cuda_dir.name} to cu{cuda_num}")
                old_cuda_dir.rename(cuda_dir)
                gpu_dockerfile = cuda_dir / "Dockerfile.gpu"
                if gpu_dockerfile.exists():
                    self._update_single_dockerfile(gpu_dockerfile, gpu_image.image_uri, "GPU")
                else:
                    self.logger.warning(f"GPU Dockerfile not found after rename: {gpu_dockerfile}")
            else:
                self.logger.warning(f"No CUDA directory found in {py3_dir}")

    def _update_single_dockerfile(self, dockerfile_path: Path, new_image_uri: str, image_type: str):
        """Update a single Dockerfile with new FROM statement and AUTOGLUON_VERSION"""
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            original_content = content
            content = re.sub(
                r'^FROM\s+.*$',
                f'FROM {new_image_uri}',
                content,
                flags=re.MULTILINE
            )
            content = re.sub(
                r'AUTOGLUON_VERSION=[\d.]+',
                f'AUTOGLUON_VERSION={self.current_version}',
                content
            )
            if content != original_content:
                with open(dockerfile_path, 'w') as f:
                    f.write(content)
                self.logger.info(f"✅ Updated {image_type} Dockerfile: {dockerfile_path.relative_to(Path('.'))}")
                self.logger.info(f"   FROM: {new_image_uri}")
            else:
                self.logger.info(f"ℹ️  No changes needed for {dockerfile_path.relative_to(Path('.'))}")
        except Exception as e:
            self.logger.error(f"❌ Failed to update {dockerfile_path}: {e}")

    def _find_latest_source_dir(self, container_type: str) -> Optional[Path]:
        """Find the latest existing docker directory to copy from"""
        base_dir = Path(f"autogluon/{container_type}/docker")
        if not base_dir.exists():
            self.logger.error(f"Base directory does not exist: {base_dir}")
            return None
        version_dirs = []
        for d in base_dir.iterdir():
            if d.is_dir():
                try:
                    version_parts = d.name.split('.')
                    if len(version_parts) >= 2:
                        version_float = float(f"{version_parts[0]}.{version_parts[1]}")
                        version_dirs.append((version_float, d))
                except ValueError:
                    continue
        if version_dirs:
            latest = max(version_dirs, key=lambda x: x[0])
            self.logger.info(f"Found latest {container_type} source: {latest[1]} (version {latest[0]})")
            return latest[1]
        self.logger.error(f"No valid version directories found in {base_dir}")
        return None

    def run_automation(self, steps_only=None):
        """Run the complete automation or specific steps"""
        original_dir = os.getcwd()
        try:
            if not steps_only or 1 in steps_only:
                self.step1_create_branch()
            if not steps_only or 2 in steps_only or 3 in steps_only:
                current_dir = Path(os.getcwd())
                self.logger.info(f"Current directory: {current_dir}")
                if not Path("autogluon").exists():
                    possible_paths = [
                        Path("deep-learning-containers"),
                        self.workspace_dir / "deep-learning-containers",
                        Path("./deep-learning-container/deep-learning-containers")
                    ]
                    found_repo = False
                    for path in possible_paths:
                        if path.exists() and (path / "autogluon").exists():
                            os.chdir(path)
                            self.logger.info(f"Changed to repo directory: {os.getcwd()}")
                            found_repo = True
                            break
                    if not found_repo:
                        self.logger.error("Could not find repository directory with autogluon folder")
                        return
                else:
                    self.logger.info(f"Already in correct directory: {os.getcwd()}")
            if not steps_only or 2 in steps_only:
                self.step2_update_toml()
            if not steps_only or 3 in steps_only:
                self.step3_create_docker_resources()
            if not steps_only or 4 in steps_only:
                self.step4_update_buildspec_files()
            if not steps_only or 5 in steps_only:
                self.step5_package_model()
            if not steps_only or 6 in steps_only:
                self.step6_build_upload_docker()
            if steps_only and len(steps_only) <= 6:
                print(f"✅ AutoGluon {self.current_version} automation completed (steps {steps_only})")
                print(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
                print(f"Branch: autogluon-{self.current_version}-release")
                print(f"Working directory: {os.getcwd()}")
                if hasattr(self, 'selected_images'):
                    print(f"Selected PyTorch: {self.selected_images.get('pytorch_version', 'N/A')}")
                    print(f"Selected CUDA: {self.selected_images.get('cuda_version', 'N/A')}")
                return
            print(f"✅ AutoGluon {self.current_version} release automation completed")
            print(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
            print(f"Branch: autogluon-{self.current_version}-release")
        except Exception as e:
            print(f"❌ Automation failed: {e}")
            self.logger.exception("Full error details:")
            raise
        finally:
            os.chdir(original_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AutoGluon DLC Release Automation')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)') 
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    parser.add_argument('--steps-only', nargs='+', type=int, help='Run only specific steps (e.g., 1 2 3)')
    args = parser.parse_args()
    automation = AutoGluonReleaseAutomation(
        args.current_version,
        args.previous_version, 
        args.fork_url
    )
    automation.run_automation(steps_only=args.steps_only)

if __name__ == "__main__":
    main()