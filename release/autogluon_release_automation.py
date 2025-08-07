"""
title : AutoGluon Release Images and Documentation Automation

description : Comprehensive two-phase automation system for updating release configuration files
and documentation. Phase 1 updates YAML release_images files with new AutoGluon
configurations and creates PR for review. Phase 2 reverts YAML changes, updates
available_images.md documentation with constructed image URLs, and creates combined
PR for final release. Handles both major and minor releases with intelligent version
shifting, ECR integration for image metadata extraction, and automated git workflows.
"""

import os
import re
import sys
import yaml
import logging
import subprocess
import shutil
import boto3
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
from automation.common import BaseAutomation, ECRImageSelector
from testing.github_pr_automation import GitHubPRAutomation
from automation.automation_logger import LoggerMixin

class AutoGluonReleaseImagesAutomation(BaseAutomation,LoggerMixin):
    """
    Automation for updating release_images files and available_images.md with AutoGluon configuration
    """
    PRODUCTION_ACCOUNT_ID = '763104351884'
    DEFAULT_REGION = 'us-west-2'
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str, yaml_only: bool = False):
        super().__init__(current_version, previous_version, fork_url)
        self.training_file = self.repo_dir / "release_images_training.yml"
        self.inference_file = self.repo_dir / "release_images_inference.yml"
        self.available_images_file = self.repo_dir / "available_images.md"
        if yaml_only:
            branch_suffix = "release"
        else:
            branch_suffix = "update"
        self.branch_name = f"{current_version}-{branch_suffix}"
        
        self.pr_automation = GitHubPRAutomation(
            current_version=current_version,
            fork_url=fork_url,
            repo_dir=self.repo_dir
        )
        self.pr_automation.branch_name = self.branch_name
        self.original_training_content = None
        self.original_inference_content = None
        self.original_available_images_content = None
        self.setup_logging(current_version,custom_name="autogluon_release")

    def get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment or GitHub CLI"""
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            return token
        try:
            result = self.run_subprocess_with_logging(
                ["gh", "auth", "token"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Could not get GitHub token from gh CLI")
            return None

    def configure_git_with_token(self, token: str) -> bool:
        """Configure Git with GitHub token for authentication"""
        try:
            self.run_subprocess_with_logging(
                ["git", "config", "user.name", "Atharva-Rajan-Kale"], 
                capture_output=True
            )
            self.run_subprocess_with_logging(
                ["git", "config", "user.email", "atharvakale912@gmail.com"], 
                capture_output=True
            )
            result = self.run_subprocess_with_logging(
                ["git", "remote", "get-url", "origin"], 
                capture_output=True, 
                text=True
            )
            current_url = result.stdout.strip()
            if "@github.com" in current_url and "https://" in current_url:
                self.logger.info("✅ Git remote already configured with token")
                return True
            if current_url.startswith("https://github.com/"):
                authenticated_url = current_url.replace(
                    "https://github.com/", 
                    f"https://{token}@github.com/"
                )
                self.run_subprocess_with_logging(
                    ["git", "remote", "set-url", "origin", authenticated_url], 
                    check=True
                )
                self.logger.info("✅ Configured git remote with GitHub token")
            else:
                self.logger.warning(f"⚠️ Unexpected remote URL format: {current_url}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to configure git with token: {e}")
            return False

    def setup_git_config(self):
        """Setup git configuration for CI environment"""
        try:
            token = self.get_github_token()
            if not token:
                self.logger.error("❌ No GitHub token available")
                return False
            if not self.configure_git_with_token(token):
                self.logger.error("❌ Failed to configure git with token")
                return False
            self.logger.info("✅ Git configuration and authentication set")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to setup git config: {e}")
            return False

    def setup_repository(self) -> bool:
        """Clone repository from fork URL (master branch) and create/checkout release branch for pushing"""
        self.logger.info("🔧 Setting up repository from fork...")
        original_dir = os.getcwd()
        try:
            self.workspace_dir.mkdir(exist_ok=True)
            os.chdir(self.workspace_dir)
            if Path("deep-learning-containers").exists():
                self.logger.info("Removing existing repository clone...")
                shutil.rmtree("deep-learning-containers")
            self.logger.info(f"Cloning from {self.fork_url} (master branch)")
            self.run_subprocess_with_logging([
                "git", "clone", self.fork_url, "deep-learning-containers"
            ], check=True)
            os.chdir("deep-learning-containers")
            if not self.setup_git_config():
                return False
            self.logger.info("🔧 Adding upstream remote for clean branch creation...")
            self.run_subprocess_with_logging([
                "git", "remote", "add", "upstream", "https://github.com/aws/deep-learning-containers.git"
            ], check=True)
            self.logger.info("🔄 Fetching upstream master for clean branch...")
            self.run_subprocess_with_logging([
                "git", "fetch", "upstream", "master"
            ], check=True)
            branch_name = self.branch_name
            self.logger.info(f"Creating clean release branch '{branch_name}' from upstream/master...")
            try:
                result = self.run_subprocess_with_logging([
                    "git", "branch", "--list", branch_name
                ], capture_output=True, text=True, check=False)
                if result.stdout.strip():
                    self.logger.info(f"Deleting existing local branch: {branch_name}")
                    self.run_subprocess_with_logging([
                        "git", "branch", "-D", branch_name
                    ], check=True)
                self.run_subprocess_with_logging([
                    "git", "checkout", "-b", branch_name, "upstream/master"
                ], check=True)
                self.logger.info(f"✅ Created clean release branch: {branch_name} from upstream/master")
                self.logger.info(f"📋 This ensures PR will only show your changes, not fork history")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to create release branch: {e}")
                return False
            self.repo_dir = Path.cwd()
            self.logger.info(f"✅ Repository setup complete: {self.repo_dir}")
            self.logger.info(f"📋 Workflow: Clean branch from upstream/master → Working on {branch_name} → Will push to fork")
            return True
        except Exception as e:
            self.logger.error(f"❌ Repository setup failed: {e}")
            return False
        finally:
            os.chdir(original_dir)

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

    def run_yaml_only_automation(self):
        """Run only YAML file updates and PR creation"""
        try:
            print("🚀 Starting AutoGluon YAML-Only Release Automation...")
            print("🔧 Setting up repository from fork...")
            if not self.setup_repository():
                raise Exception("Failed to setup repository from fork")
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
            print(f"🔄 Committing and pushing changes: {commit_message}")
            if not self.commit_and_push_yaml_changes(commit_message):
                raise Exception("Failed to commit and push YAML changes")
            print("✅ Changes committed and pushed successfully")
            print("🚀 Creating Pull Request for YAML changes...")
            self.logger.info(f"Using branch name: {self.pr_automation.branch_name}")
            original_dir = os.getcwd()
            try:
                os.chdir(self.repo_dir)
                self.logger.info("🔧 Resetting remote URL for PR automation...")
                clean_url = self.fork_url
                self.run_subprocess_with_logging([
                    "git", "remote", "set-url", "origin", clean_url
                ], check=True)
                if not self.pr_automation.create_pull_request():
                    raise Exception("Failed to create Pull Request")
                print("✅ Pull Request created successfully")
            finally:
                os.chdir(original_dir)
            print("✅ YAML-Only AutoGluon Release Automation completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ YAML-only automation failed: {e}")
            print(f"❌ Error: {e}")
            return False

    def run_revert_and_available_images_automation(self):
        """Run revert YAML + update available_images.md + create single combined PR"""
        try:
            print("🚀 Starting AutoGluon Revert and Available Images Automation...")
            
            print("🔧 Setting up repository from fork...")
            if not self.setup_repository():
                raise Exception("Failed to setup repository from fork")
            
            print("🔍 Extracting image information for available_images.md...")
            image_info = self.get_latest_training_gpu_image()
            if not image_info:
                raise Exception("Failed to get training image information")
            print(f"📦 Using image info from {image_info['tag']} for URL construction")            
            print("🔄 Reverting YAML changes...")
            if not self.revert_yaml_files():
                raise Exception("Failed to revert YAML files")
            print("✅ YAML files reverted successfully")
            print("📝 Starting available_images.md update process...")
            print("🔧 Constructing image URLs using specified pattern...")
            print(f"📋 Using production account ID: {self.PRODUCTION_ACCOUNT_ID}")
            print(f"📋 Using region: {os.environ.get('REGION', self.DEFAULT_REGION)}")
            constructed_images = self.construct_image_urls_by_type(image_info)
            training_images = constructed_images['training']
            inference_images = constructed_images['inference']
            if not training_images or not inference_images:
                raise Exception("Failed to construct sufficient image URLs")
            print(f"📦 Constructed training images: {list(training_images.keys()) if isinstance(training_images, dict) else type(training_images)}")
            print(f"📦 Constructed inference images: {list(inference_images.keys()) if isinstance(inference_images, dict) else type(inference_images)}")
            for img_type, img_data in training_images.items():
                self.logger.info(f"📦 Training {img_type} constructed URL: {img_data.get('image_uri', 'N/A')}")
            for img_type, img_data in inference_images.items():
                self.logger.info(f"📦 Inference {img_type} constructed URL: {img_data.get('image_uri', 'N/A')}")
            print("📝 Updating available_images.md with constructed image URLs...")
            if not self.update_available_images_md(training_images, inference_images):
                raise Exception("Failed to update available_images.md")
            print("✅ available_images.md updated successfully with constructed image URLs")
            if not self.prompt_user("Commit and create PR with both YAML revert and available_images.md changes?"):
                print("❌ Operation cancelled by user")
                return False
            combined_commit_message = f"AutoGluon {self.current_version}: Revert release images config and update available images documentation"
            print(f"🔄 Committing and pushing combined changes: {combined_commit_message}")
            if not self.commit_and_push_combined_changes(combined_commit_message):
                raise Exception("Failed to commit and push combined changes")
            print("✅ Combined changes committed and pushed successfully")
            print("🚀 Creating Pull Request for combined changes...")
            self.logger.info(f"Using branch name: {self.pr_automation.branch_name}")
            original_dir = os.getcwd()
            try:
                os.chdir(self.repo_dir)
                self.logger.info("🔧 Resetting remote URL for PR automation...")
                clean_url = self.fork_url
                self.run_subprocess_with_logging([
                    "git", "remote", "set-url", "origin", clean_url
                ], check=True)
                if not self.pr_automation.create_pull_request():
                    raise Exception("Failed to create Pull Request")
                print("✅ Pull Request created successfully")
            finally:
                os.chdir(original_dir)
            print("✅ Revert and Available Images Automation completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Revert and available images automation failed: {e}")
            print(f"❌ Error: {e}")
            return False

    def commit_and_push_combined_changes(self, commit_message: str):
        """Commit and push both YAML revert and available_images.md changes in single commit"""
        try:
            original_dir = os.getcwd()
            os.chdir(self.repo_dir)
            self.run_subprocess_with_logging(['git', 'add', 'release_images_training.yml', 'release_images_inference.yml', 'available_images.md'], check=True)
            self.run_subprocess_with_logging(['git', 'commit', '-m', commit_message], check=True)
            self.run_subprocess_with_logging(['git', 'push', 'origin', self.branch_name], check=True)
            self.logger.info(f"✅ Combined changes committed and pushed: {commit_message}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to commit and push combined changes: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def run_release_images_automation(self, yaml_only: bool = False):
        """Main automation workflow dispatcher"""
        if yaml_only:
            return self.run_yaml_only_automation()
        else:
            return self.run_revert_and_available_images_automation()

    def construct_image_urls_by_type(self, reference_image_info: Dict) -> Dict[str, Dict]:
        """Construct image URLs for both training and inference, GPU and CPU using the specified pattern"""
        self.logger.info("🔧 Constructing image URLs using specified pattern...")
        account_id = self.PRODUCTION_ACCOUNT_ID
        python_version = reference_image_info['python_versions'][0]
        cuda_version = reference_image_info['cuda_version']
        os_version = reference_image_info['os_version']
        self.logger.info(f"📋 Using extracted info - Python: {python_version}, CUDA: {cuda_version}, OS: {os_version}")
        constructed_images = {
            'training': {},
            'inference': {}
        }
        for job_type in ['training', 'inference']:
            # GPU image
            gpu_tag = f"{self.current_version}-gpu-{python_version}-{cuda_version}-{os_version}"
            gpu_uri = f"{account_id}.dkr.ecr.us-west-2.amazonaws.com/autogluon-{job_type}:{gpu_tag}"
            constructed_images[job_type]['gpu'] = {
                'image_uri': gpu_uri,
                'tag': gpu_tag,
                'python_versions': [python_version],
                'cuda_version': cuda_version,
                'os_version': os_version
            }
            # CPU image (no CUDA version)
            cpu_tag = f"{self.current_version}-cpu-{python_version}-{os_version}"
            cpu_uri = f"{account_id}.dkr.ecr.us-west-2.amazonaws.com/autogluon-{job_type}:{cpu_tag}"
            constructed_images[job_type]['cpu'] = {
                'image_uri': cpu_uri,
                'tag': cpu_tag,
                'python_versions': [python_version],
                'cuda_version': '',
                'os_version': os_version
            }
            self.logger.info(f"📦 Constructed {job_type} GPU URL: {gpu_uri}")
            self.logger.info(f"📦 Constructed {job_type} CPU URL: {cpu_uri}")
        return constructed_images
        
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
        """Update available_images.md with new AutoGluon version information using constructed URLs"""
        try:
            self.logger.info("📝 Updating available_images.md with constructed URLs...")
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
                self.logger.error("❌ GPU images not found in training or inference images")
                return False
            # Process the content based on whether it's a major release
            if self.is_major_release:
                self.logger.info("🔄 Major version update detected - moving current AutoGluon sections to Prior sections")
                content = self.move_current_autogluon_to_prior(content)
            else:
                self.logger.info("🔄 Minor version update detected - keeping Prior sections unchanged, updating current sections only")
            # Always update main sections with new version
            content = self.update_autogluon_sections(content, training_images, inference_images)
            self.logger.info(f"📋 Updated file size: {len(content)} characters")
            with open(self.available_images_file, 'w') as f:
                f.write(content)
            self.logger.info("✅ available_images.md updated successfully with constructed URLs")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update available_images.md: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        
    def create_current_inference_section(self, inference_images: Dict[str, Dict]) -> list[str]:
        """Create the current AutoGluon Inference Containers section"""
        section_lines = [
            "AutoGluon Inference Containers",
            "===============================",
            "",
            "| Framework       | AutoGluon Version  | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |",
            "|-----------------|--------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|"
        ]

        if 'gpu' in inference_images:
            gpu_row = self.create_table_row_with_constructed_url(inference_images['gpu'], "inference", "GPU")
            section_lines.append(gpu_row)
        if 'cpu' in inference_images:
            cpu_row = self.create_table_row_with_constructed_url(inference_images['cpu'], "inference", "CPU")
            section_lines.append(cpu_row)
        section_lines.append("")
        return section_lines
    
    def create_current_training_section(self, training_images: Dict[str, Dict]) -> list[str]:
        """Create the current AutoGluon Training Containers section"""
        section_lines = [
            "AutoGluon Training Containers",
            "===============================", 
            "",
            "| Framework       | AutoGluon Version  | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |",
            "|-----------------|--------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|"
        ]
        if 'gpu' in training_images:
            gpu_row = self.create_table_row_with_constructed_url(training_images['gpu'], "training", "GPU")
            section_lines.append(gpu_row)
        if 'cpu' in training_images:
            cpu_row = self.create_table_row_with_constructed_url(training_images['cpu'], "training", "CPU")
            section_lines.append(cpu_row)
        section_lines.append("")
        return section_lines
    
    def update_autogluon_sections(self, content: str, training_images: Dict[str, Dict], inference_images: Dict[str, Dict]) -> str:
        """Update the main AutoGluon sections with new version information"""
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Handle AutoGluon Training Containers section
            if line == "AutoGluon Training Containers":
                new_lines.extend(self.create_current_training_section(training_images))
                i = self.skip_to_next_section(lines, i)
                continue
            # Handle AutoGluon Inference Containers section
            elif line == "AutoGluon Inference Containers":
                new_lines.extend(self.create_current_inference_section(inference_images))
                i = self.skip_to_next_section(lines, i)
                continue
            else:
                new_lines.append(lines[i])
                i += 1
        return '\n'.join(new_lines)
    
    def create_prior_training_section(self, training_rows: list[str]) -> list[str]:
        """Create the Prior AutoGluon Training Containers section"""
        section_lines = [
            "Prior AutoGluon Training Containers",
            "===============================",
            "",
            "| Framework       | AutoGluon Version | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |",
            "|-----------------|-------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|"
        ]
        if training_rows:
            section_lines.extend(training_rows)
        section_lines.append("")
        return section_lines

    def create_prior_inference_section(self, inference_rows: list[str]) -> list[str]:
        """Create the Prior AutoGluon Inference Containers section"""
        section_lines = [
            "Prior AutoGluon Inference Containers", 
            "===============================",
            "",
            "| Framework       | AutoGluon Version | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |",
            "|-----------------|-------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|"
        ]
        if inference_rows:
            section_lines.extend(inference_rows)
        section_lines.append("")
        return section_lines

    def skip_to_next_section(self, lines: list[str], current_idx: int) -> int:
        """Skip lines until the next major section is found"""
        i = current_idx + 1
        while i < len(lines):
            line = lines[i].strip()
            if (line.endswith(" Containers") and 
                i + 1 < len(lines) and 
                lines[i + 1].strip().startswith("=")):
                return i
            i += 1
        return len(lines)
    
    def move_current_autogluon_to_prior(self, content: str) -> str:
        """Move current AutoGluon sections to Prior sections for major version updates"""
        lines = content.split('\n')
        # Extract current sections that will become the new prior sections
        current_training_rows = self.extract_autogluon_table_rows(lines, "AutoGluon Training Containers")
        current_inference_rows = self.extract_autogluon_table_rows(lines, "AutoGluon Inference Containers")
        self.logger.info(f"📋 Moving {len(current_training_rows)} training rows and {len(current_inference_rows)} inference rows to Prior sections")
        # Rebuild the content with updated prior sections
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Handle Prior AutoGluon Training Containers section
            if line == "Prior AutoGluon Training Containers":
                new_lines.extend(self.create_prior_training_section(current_training_rows))
                i = self.skip_to_next_section(lines, i)
                continue
            # Handle Prior AutoGluon Inference Containers section  
            elif line == "Prior AutoGluon Inference Containers":
                new_lines.extend(self.create_prior_inference_section(current_inference_rows))
                i = self.skip_to_next_section(lines, i)
                continue
            else:
                new_lines.append(lines[i])
                i += 1
        content_str = '\n'.join(new_lines)
        if "Prior AutoGluon Training Containers" not in content_str:
            content_str = self.insert_prior_sections(content_str, current_training_rows, current_inference_rows)
        return content_str
    
    def insert_prior_sections(self, content: str, training_rows: list[str], inference_rows: list[str]) -> str:
        """Insert Prior AutoGluon sections if they don't exist"""
        lines = content.split('\n')
        insert_idx = len(lines)
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped in ["HuggingFace Training Containers", 
                            "StabilityAI Inference Containers",
                            "SageMaker Training Compiler Containers"]:
                insert_idx = i
                break
        # Create both prior sections
        prior_sections = []
        if training_rows:
            prior_sections.extend(self.create_prior_training_section(training_rows))
        if inference_rows:
            prior_sections.extend(self.create_prior_inference_section(inference_rows))
        # Insert the sections
        for idx, section_line in enumerate(prior_sections):
            lines.insert(insert_idx + idx, section_line)
        return '\n'.join(lines)
    
    def create_table_row_with_constructed_url(self, image_info: Dict, job_type: str, compute_type: str) -> str:
        """Create a table row using the constructed image URL"""
        python_version = self.convert_python_version(image_info['python_versions'][0])
        constructed_url = image_info['image_uri']
        self.logger.info(f"📋 Creating {job_type} {compute_type} row with constructed URL: {constructed_url}")
        if job_type == "training":
            return f"| AutoGluon {self.current_version} | {self.current_version}              | {job_type} | {compute_type}     | {python_version} ({image_info['python_versions'][0]})             | {constructed_url}       |"
        else:  # inference
            return f"| AutoGluon {self.current_version} | {self.current_version}              | {job_type} | {compute_type}     | {python_version} ({image_info['python_versions'][0]})             | {constructed_url}       |"    
    
    def find_section_boundaries(self, lines: list[str], section_name: str) -> tuple[int, int]:
        """Find the start and end line indices of a section"""
        start_idx = -1
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == section_name:
                start_idx = i
                break
        if start_idx == -1:
            return -1, -1
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if (line.endswith(" Containers") and 
                i + 1 < len(lines) and 
                lines[i + 1].strip().startswith("=")):
                end_idx = i
                break
        return start_idx, end_idx
    
    def extract_autogluon_table_rows(self, lines: list[str], section_name: str) -> list[str]:
        """Extract only the AutoGluon table rows from a section"""
        start_idx, end_idx = self.find_section_boundaries(lines, section_name)
        if start_idx == -1:
            self.logger.warning(f"⚠️ Section '{section_name}' not found")
            return []
        table_rows = []
        in_table = False
        for i in range(start_idx, end_idx):
            line = lines[i].strip()
            if line.startswith("|--"):
                in_table = True
                continue
            if in_table and line.startswith("|") and "AutoGluon" in line:
                table_rows.append(lines[i]) 
            elif in_table and line and not line.startswith("|"):
                break
        self.logger.info(f"📋 Extracted {len(table_rows)} table rows from '{section_name}'")
        return table_rows
            
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
            
    def remove_autogluon_from_yaml(self, yaml_file: Path):
        """Remove AutoGluon configurations from a specific YAML file"""
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
            lines = content.split('\n')
            new_lines = []
            skip_section = False
            i = 0
            while i < len(lines):
                line = lines[i]
                if re.match(r'\s*\d+:\s*$', line):
                    lookahead_lines = []
                    j = i
                    while j < len(lines) and j < i + 10: 
                        lookahead_lines.append(lines[j])
                        if 'framework: "autogluon"' in lines[j]:
                            skip_section = True
                            self.logger.info(f"🗑️ Found AutoGluon section starting at line {i+1}, removing...")
                            break
                        elif re.match(r'\s*\d+:\s*$', lines[j]) and j > i:
                            break
                        j += 1
                    if skip_section:
                        while i < len(lines):
                            current_line = lines[i]
                            i += 1
                            if (i < len(lines) and 
                                re.match(r'\s*\d+:\s*$', lines[i]) and 
                                not lines[i].startswith('  ')):
                                break
                        skip_section = False
                        continue
                    else:
                        new_lines.append(line)
                        i += 1
                else:
                    new_lines.append(line)
                    i += 1
            new_content = '\n'.join(new_lines)
            with open(yaml_file, 'w') as f:
                f.write(new_content)
            self.logger.info(f"✅ Removed AutoGluon configurations from {yaml_file.name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to remove AutoGluon from {yaml_file}: {e}")
            return False
        
    def revert_yaml_files(self):
        """Remove AutoGluon configurations from YAML files"""
        try:
            self.logger.info("🔄 Removing AutoGluon configurations from YAML files...")
            # Revert training file
            if not self.remove_autogluon_from_yaml(self.training_file):
                return False
            # Revert inference file  
            if not self.remove_autogluon_from_yaml(self.inference_file):
                return False
            self.logger.info("✅ AutoGluon configurations removed from YAML files")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to revert YAML files: {e}")
            return False
            
    def commit_and_push_yaml_changes(self, commit_message: str):
        """Commit and push YAML file changes to the branch"""
        try:
            original_dir = os.getcwd()
            os.chdir(self.repo_dir)
            self.run_subprocess_with_logging(['git', 'add', 'release_images_training.yml', 'release_images_inference.yml'], check=True)
            self.run_subprocess_with_logging(['git', 'commit', '-m', commit_message], check=True)
            self.run_subprocess_with_logging(['git', 'push', 'origin', self.branch_name], check=True)
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
            self.run_subprocess_with_logging(['git', 'add', 'available_images.md'], check=True)
            # Commit changes
            self.run_subprocess_with_logging(['git', 'commit', '-m', commit_message], check=True)
            # Push changes
            branch_name = f"{self.current_version}-release"
            self.run_subprocess_with_logging(['git', 'push', 'origin', branch_name], check=True)
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

def main():
    """Main function for AutoGluon Release Images Automation"""
    import argparse
    parser = argparse.ArgumentParser(description='AutoGluon Release Images Automation - Updates YAML files and available_images.md')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    parser.add_argument('--yaml-only', action='store_true', help='Run only YAML file updates and PR creation (first step). Without this flag, runs revert + available_images.md update (second step).')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    if args.yaml_only:
        print("🚀 AutoGluon YAML-Only Release Automation")
        print("📋 This will update YAML files and create PR")
        print(f"📋 Branch name: {args.current_version}-release")
    else:
        print("🚀 AutoGluon Revert + Available Images Automation") 
        print("📋 This will revert YAML files, update available_images.md, and create combined PR")
        print(f"📋 Branch name: {args.current_version}-update")
        
    automation = AutoGluonReleaseImagesAutomation(
        args.current_version,
        args.previous_version,
        args.fork_url,
        yaml_only=args.yaml_only
    )
    success = automation.run_release_images_automation(yaml_only=args.yaml_only)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()