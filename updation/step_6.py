"""
title : Docker Build and ECR Upload Automation

description : Handles the complete Docker image building and uploading workflow for AutoGluon
containers. Fixes known code issues, sets up build environment with ECR authentication,
and builds all image variants (training/inference × CPU/GPU) with real-time output
streaming. Manages Python virtual environment setup, dependency installation,
and proper cleanup of build artifacts. Includes comprehensive error handling,
timeout management, and validation of required environment variables. Essential
final step that produces the actual Docker images ready for testing and deployment
to AWS ECR repositories for both beta-autogluon-training and beta-autogluon-inference.
"""

import os
import re
import shutil
import subprocess
import glob
from pathlib import Path
from automation.common import BaseAutomation
from automation.automation_logger import LoggerMixin

class Step6Automation(BaseAutomation, LoggerMixin):
    """Handles Step 6: Docker Build and Upload"""

    def __init__(self, current_version, previous_version, fork_url):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_logging(current_version, custom_name="step_6")
        
    def cleanup_build_artifacts(self):
        """Clean up docker build artifacts and unwanted files"""        
        try:
            removed_count = 0            
            specific_files = [
                "autogluon/training/out.py",
                "autogluon/training/telemetry.sh",
                "autogluon/inference/out.py",
                "autogluon/inference/telemetry.sh",
            ]
            
            for file_path in specific_files:
                full_path = Path(file_path)
                if full_path.exists():
                    try:
                        full_path.unlink()
                        self.logger.info(f"🗑️ Removed file: {file_path}")
                        removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not remove {file_path}: {e}")
            
            # Remove all .json files from src/ directory
            src_json_files = glob.glob("src/**/*.json", recursive=True)
            for json_file in src_json_files:
                try:
                    os.remove(json_file)
                    self.logger.info(f"🗑️ Removed JSON file: {json_file}")
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not remove {json_file}: {e}")
            self.logger.info(f"✅ Cleanup completed. Removed {removed_count} artifacts.")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return False
        
    def step6_build_upload_docker(self):
        """Step 6: Fix code issues, build and upload Docker images to ECR"""
        self.logger.info("Step 6: Building and uploading Docker images")

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
                self.logger.error(f"Repository directory not found: {self.repo_dir}")
                return False

            os.chdir(self.repo_dir)
            self.logger.info(f"Changed to repo directory: {os.getcwd()}")

            if not self.fix_code_issues():
                return False

            if not self.setup_build_environment(account_id, region):
                return False

            if not self.build_docker_images():
                return False
            if not self.cleanup_build_artifacts():
                self.logger.warning("⚠️ Cleanup failed, but continuing...")

            self.logger.info("✅ Step 6 completed: Docker images built and uploaded")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step 6 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
            self.logger.info(f"Returned to original directory: {os.getcwd()}")

    def fix_code_issues(self):
        """Fix known issues in the codebase"""
        self.logger.info("🔧 Fixing code issues...")
        try:
            # Fix patch_helper.py import
            patch_helper_path = Path("src/patch_helper.py")
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

            # Fix utils.py path issue
            utils_path = Path("src/utils.py")
            if utils_path.exists():
                with open(utils_path, 'r') as f:
                    content = f.read()

                if "_project_root" not in content:
                    lines = content.split('\n')
                    sys_import_index = -1
                    for i, line in enumerate(lines):
                        if line.strip() == "import sys":
                            sys_import_index = i
                            break

                    if sys_import_index != -1:
                        new_lines = [
                            "_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
                            "if _project_root not in sys.path:",
                            "    sys.path.insert(0, _project_root)"
                        ]
                        for j, new_line in enumerate(new_lines):
                            lines.insert(sys_import_index + 1 + j, new_line)

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

    def setup_build_environment(self, account_id: str, region: str):
        """Set up environment variables and dependencies"""
        self.logger.info("🛠️  Setting up build environment...")
        try:
            self.logger.info(f"Using ACCOUNT_ID={account_id}, REGION={region}")
            # ECR login for user's account
            self.logger.info("🔐 Logging into your ECR...")
            login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
            self.run_subprocess_with_logging(login_cmd, shell=True)
            self.logger.info("✅ Logged into your ECR")

            # ECR login for AWS's account
            self.logger.info("🔐 Logging into AWS's ECR...")
            aws_login_cmd = f"aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com"
            self.run_subprocess_with_logging(aws_login_cmd, shell=True)
            self.logger.info("✅ Logged into AWS's ECR")

            # Setup Python environment
            self.logger.info("🐍 Setting up Python environment...")
            if Path("dlc").exists():
                shutil.rmtree("dlc")
            self.run_subprocess_with_logging(["python3", "-m", "venv", "dlc"])
            self.run_subprocess_with_logging(["dlc/bin/pip", "install", "-U", "pip"])
            self.run_subprocess_with_logging(["dlc/bin/pip", "install", "-U", "setuptools", "wheel"])
            self.run_subprocess_with_logging(["dlc/bin/pip", "install", "-r", "src/requirements.txt"])
            self.logger.info("✅ Python dependencies installed")

            self.run_subprocess_with_logging(["dlc/bin/pip", "install", "requests==2.31.0"])
            self.logger.info("✅ Python dependencies and requests version installed")

            # Run setup script
            self.logger.info("🔧 Running setup script...")
            self.run_subprocess_with_logging(["bash", "src/setup.sh", "autogluon"])
            self.logger.info("✅ Setup script completed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to setup build environment: {e}")
            return False

    def build_docker_images(self):
        """Build and upload all Docker images with real-time output"""
        self.logger.info("🏗️  Building Docker images...")
        try:
            # Training images
            os.environ['REPOSITORY_NAME'] = 'beta-autogluon-training'
            
            self.logger.info("Building training CPU image...")
            self.run_subprocess_with_logging([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/training/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "training",
                "--device_types", "cpu",
                "--py_versions", "py3"
            ], step_description="Building AutoGluon Training CPU Image", capture_output=False)
            self.logger.info("✅ Training CPU image built")
            
            self.logger.info("Building training GPU image...")
            self.run_subprocess_with_logging([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/training/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "training",
                "--device_types", "gpu",
                "--py_versions", "py3"
            ], step_description="Building AutoGluon Training GPU Image", capture_output=False)
            self.logger.info("✅ Training GPU image built")
            
            # Inference images
            os.environ['REPOSITORY_NAME'] = 'beta-autogluon-inference'
            
            self.logger.info("Building inference CPU image...")
            self.run_subprocess_with_logging([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/inference/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "inference",
                "--device_types", "cpu",
                "--py_versions", "py3"
            ], step_description="Building AutoGluon Inference CPU Image", capture_output=False)
            self.logger.info("✅ Inference CPU image built")
            
            self.logger.info("Building inference GPU image...")
            self.run_subprocess_with_logging([
                "dlc/bin/python", "src/main.py",
                "--buildspec", "autogluon/inference/buildspec.yml",
                "--framework", "autogluon",
                "--image_types", "inference",
                "--device_types", "gpu",
                "--py_versions", "py3"
            ], step_description="Building AutoGluon Inference GPU Image", capture_output=False)
            self.logger.info("✅ Inference GPU image built")
            
            self.logger.info("🎉 All Docker images built and uploaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to build Docker images: {e}")

            return False

    def run_steps(self, steps_only=None):
        """Run step 6"""
        results = {}
        if not steps_only or 6 in steps_only:
            results[6] = self.step6_build_upload_docker()
        return results