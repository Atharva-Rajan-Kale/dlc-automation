import os
import re
import shutil
import subprocess
from pathlib import Path
from common import BaseAutomation

class Step6Automation(BaseAutomation):
    """Handles Step 6: Docker Build and Upload"""
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
            if not self._fix_code_issues():
                return False
            if not self._setup_build_environment(account_id, region):
                return False
            if not self._build_docker_images():
                return False
            self.logger.info("✅ Step 6 completed: Docker images built and uploaded")
            return True
        except Exception as e:
            self.logger.error(f"❌ Step 6 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
            self.logger.info(f"Returned to original directory: {os.getcwd()}")

    def _fix_code_issues(self):
        """Fix known issues in the codebase"""
        self.logger.info("🔧 Fixing code issues...")
        try:
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

    def _setup_build_environment(self, account_id: str, region: str):
        """Set up environment variables and dependencies"""
        self.logger.info("🛠️  Setting up build environment...")
        try:
            self.logger.info(f"Using ACCOUNT_ID={account_id}, REGION={region}")
            self.logger.info("🔐 Logging into your ECR...")
            login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
            subprocess.run(login_cmd, shell=True, check=True)
            self.logger.info("✅ Logged into your ECR")
            self.logger.info("🔐 Logging into AWS's ECR...")
            aws_login_cmd = f"aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com"
            subprocess.run(aws_login_cmd, shell=True, check=True)
            self.logger.info("✅ Logged into AWS's ECR")
            self.logger.info("🐍 Setting up Python environment...")
            if Path("dlc").exists():
                shutil.rmtree("dlc")
            subprocess.run(["python3", "-m", "venv", "dlc"], check=True)
            subprocess.run(["dlc/bin/pip", "install", "-U", "pip"], check=True)
            subprocess.run(["dlc/bin/pip", "install", "-U", "setuptools", "wheel"], check=True)
            subprocess.run(["dlc/bin/pip", "install", "-r", "src/requirements.txt"], check=True)
            self.logger.info("✅ Python dependencies installed")
            subprocess.run(["dlc/bin/pip", "install", "requests==2.31.0"], check=True)
            self.logger.info("✅ Python dependencies and requests version installed")
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

    def run_steps(self, steps_only=None):
        """Run step 6"""
        results = {}
        if not steps_only or 6 in steps_only:
            results[6] = self.step6_build_upload_docker()
        return results