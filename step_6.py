import os
import re
import shutil
import subprocess
from pathlib import Path
from common import BaseAutomation
from automation_logger import LoggerMixin

class Step6Automation(BaseAutomation, LoggerMixin):
    """Handles Step 6: Docker Build and Upload"""
    
    def __init__(self, current_version, previous_version, fork_url):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_logging(current_version, custom_name="step_6")

    def docker_cleanup_between_builds(self):
        """Clean Docker layers between builds to prevent space issues"""
        self.logger.info("🧹 Cleaning Docker layers between builds...")
        
        # First, try standard cleanup
        cleanup_commands = [
            ['docker', 'container', 'prune', '-f'],
            ['docker', 'network', 'prune', '-f'],
            ['docker', 'image', 'prune', '-f'],
            ['docker', 'builder', 'prune', '-f']
        ]
        
        for cmd in cleanup_commands:
            try:
                result = self.run_subprocess_with_logging(cmd, capture_output=True, text=True, check=False)
                if result.stdout and result.stdout.strip():
                    self.logger.info(f"Cleanup output: {result.stdout.strip()}")
            except Exception as e:
                self.logger.warning(f"Cleanup command failed: {' '.join(cmd)} - {e}")
        
        # Check if we still have reclaimable space
        try:
            result = self.run_subprocess_with_logging(['docker', 'system', 'df'], capture_output=True, text=True, check=False)
            self.logger.info(f"Docker space after basic cleanup:\n{result.stdout}")
            
            # Parse the output to see if we have reclaimable space
            if result.stdout and "RECLAIMABLE" in result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Images" in line and "GB" in line and "(" in line:
                        # Extract reclaimable space percentage
                        if "%" in line:
                            parts = line.split("(")
                            if len(parts) > 1:
                                percent_part = parts[1].split("%")[0]
                                try:
                                    reclaimable_percent = int(percent_part)
                                    if reclaimable_percent > 50:  # If more than 50% is reclaimable
                                        self.logger.warning(f"🚨 {reclaimable_percent}% Docker space is reclaimable - performing aggressive cleanup")
                                        self.aggressive_image_cleanup()
                                        break
                                except:
                                    pass
        except Exception as e:
            self.logger.warning(f"Could not check Docker space: {e}")

    def aggressive_image_cleanup(self):
        """Aggressively remove Docker images to free space"""
        self.logger.warning("🚨 Performing aggressive Docker image cleanup...")
        
        try:
            # Get list of all images except base ones
            result = self.run_subprocess_with_logging(['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'], 
                                                     capture_output=True, text=True, check=False)
            
            if result.stdout:
                all_images = result.stdout.strip().split('\n')
                images_to_keep = [
                    'ubuntu', 'python', 'nvidia/cuda', 'amazon/aws-cli'
                ]
                
                images_to_remove = []
                for image in all_images:
                    if image and image != '<none>:<none>':
                        # Keep base images but remove AutoGluon builds
                        should_keep = False
                        for keep_pattern in images_to_keep:
                            if keep_pattern in image.lower():
                                should_keep = True
                                break
                        
                        if not should_keep:
                            images_to_remove.append(image)
                
                # Remove images in batches
                if images_to_remove:
                    self.logger.info(f"🗑️ Removing {len(images_to_remove)} Docker images to free space")
                    
                    # Remove images one by one to avoid command line length issues
                    for image in images_to_remove:
                        try:
                            self.run_subprocess_with_logging(['docker', 'rmi', '-f', image], 
                                                           capture_output=True, text=True, check=False)
                        except:
                            pass  # Continue even if some images fail to remove
                    
                    self.logger.info("✅ Aggressive image cleanup completed")
                
            # Final cleanup
            self.run_subprocess_with_logging(['docker', 'system', 'prune', '-af'], check=False)
            
            # Show final space
            result = self.run_subprocess_with_logging(['docker', 'system', 'df'], capture_output=True, text=True, check=False)
            self.logger.info(f"Docker space after aggressive cleanup:\n{result.stdout}")
            
        except Exception as e:
            self.logger.error(f"❌ Aggressive image cleanup failed: {e}")

    def check_disk_space(self, stage=""):
        """Check and log current disk space"""
        try:
            result = self.run_subprocess_with_logging(['df', '-h', '/'], capture_output=True, text=True, check=False)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    header = lines[0]
                    data = lines[1]
                    self.logger.info(f"💾 Disk space {stage}:\n{header}\n{data}")
                    
                    # Extract usage percentage
                    fields = data.split()
                    if len(fields) >= 5:
                        usage_percent = fields[4].rstrip('%')
                        if usage_percent.isdigit() and int(usage_percent) > 85:
                            self.logger.warning(f"⚠️ High disk usage: {usage_percent}%")
                            return int(usage_percent)
            return 0
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return 0

    def aggressive_docker_cleanup(self):
        """Perform aggressive Docker cleanup when space is critically low"""
        self.logger.warning("🚨 Performing aggressive Docker cleanup due to low space...")
        
        try:
            # Stop all containers
            result = self.run_subprocess_with_logging(['docker', 'ps', '-q'], capture_output=True, text=True, check=False)
            if result.stdout and result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                self.run_subprocess_with_logging(['docker', 'stop'] + container_ids, check=False)
                self.logger.info("🛑 Stopped all running containers")
            
            # Remove all containers
            result = self.run_subprocess_with_logging(['docker', 'ps', '-aq'], capture_output=True, text=True, check=False)
            if result.stdout and result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                self.run_subprocess_with_logging(['docker', 'rm', '-f'] + container_ids, check=False)
                self.logger.info("🗑️ Removed all containers")
            
            # Remove all images except base ones we might need
            self.run_subprocess_with_logging(['docker', 'system', 'prune', '-af', '--volumes'], check=False)
            self.logger.info("🧹 Aggressive system cleanup completed")
            
            # Check space after cleanup
            self.check_disk_space("after aggressive cleanup")
            
        except Exception as e:
            self.logger.error(f"❌ Aggressive cleanup failed: {e}")

    def build_single_image_safely(self, build_args, description):
        """Build a single Docker image with proper error handling and cleanup"""
        self.logger.info(f"🐳 Starting: {description}")
        
        # Check space before build
        usage = self.check_disk_space(f"before {description}")
        if usage > 90:
            self.logger.warning(f"⚠️ Very low disk space ({usage}%) - performing aggressive cleanup")
            self.aggressive_docker_cleanup()
        elif usage > 80:
            self.logger.warning(f"⚠️ Low disk space ({usage}%) - performing cleanup")
            self.docker_cleanup_between_builds()
        
        try:
            # Build the image - NO TIMEOUT parameter to avoid Popen error
            self.run_subprocess_with_logging(
                build_args,
                step_description=description,
                capture_output=False
                # ✅ NO timeout parameter here - this was causing the Popen error
            )
            
            self.logger.info(f"✅ {description} completed successfully")
            
            # Immediate cleanup after successful build
            self.docker_cleanup_between_builds()
            
            # Check space after build
            self.check_disk_space(f"after {description}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ {description} failed with exit code {e.returncode}")
            self.docker_cleanup_between_builds()
            return False
        except Exception as e:
            self.logger.error(f"❌ {description} failed: {e}")
            self.docker_cleanup_between_builds()
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
        
        # Initial space check
        self.check_disk_space("before setup")
        
        try:
            self.logger.info(f"Using ACCOUNT_ID={account_id}, REGION={region}")
            
            # Initial Docker cleanup to start fresh
            self.logger.info("🧹 Initial Docker cleanup...")
            self.docker_cleanup_between_builds()
            
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
            
            # Check space after setup
            self.check_disk_space("after setup")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to setup build environment: {e}")
            return False

    def build_docker_images(self):
        """Build and upload all Docker images with real-time output and space management"""
        self.logger.info("🏗️  Building Docker images with space management...")
        
        # Initial space check and cleanup
        self.check_disk_space("before Docker builds")
        self.docker_cleanup_between_builds()
        
        try:
            images_to_build = [
                {
                    'repo_name': 'beta-autogluon-training',
                    'buildspec': 'autogluon/training/buildspec.yml',
                    'image_type': 'training',
                    'device': 'cpu',
                    'description': 'Training CPU Image'
                },
                {
                    'repo_name': 'beta-autogluon-training', 
                    'buildspec': 'autogluon/training/buildspec.yml',
                    'image_type': 'training',
                    'device': 'gpu',
                    'description': 'Training GPU Image'
                },
                {
                    'repo_name': 'beta-autogluon-inference',
                    'buildspec': 'autogluon/inference/buildspec.yml', 
                    'image_type': 'inference',
                    'device': 'cpu',
                    'description': 'Inference CPU Image'
                },
                {
                    'repo_name': 'beta-autogluon-inference',
                    'buildspec': 'autogluon/inference/buildspec.yml',
                    'image_type': 'inference', 
                    'device': 'gpu',
                    'description': 'Inference GPU Image'
                }
            ]
            
            successful_builds = []
            failed_builds = []
            
            for i, config in enumerate(images_to_build, 1):
                self.logger.info(f"📦 Building image {i}/{len(images_to_build)}: {config['description']}")
                
                # Set repository name environment variable
                os.environ['REPOSITORY_NAME'] = config['repo_name']
                
                # Prepare build arguments
                build_args = [
                    "dlc/bin/python", "src/main.py",
                    "--buildspec", config['buildspec'],
                    "--framework", "autogluon",
                    "--image_types", config['image_type'],
                    "--device_types", config['device'],
                    "--py_versions", "py3"
                ]
                
                # Build the image with safety measures
                success = self.build_single_image_safely(build_args, config['description'])
                
                if success:
                    successful_builds.append(config['description'])
                    self.logger.info(f"✅ Completed {i}/{len(images_to_build)}: {config['description']}")
                else:
                    failed_builds.append(config['description'])
                    self.logger.error(f"❌ Failed {i}/{len(images_to_build)}: {config['description']}")
                    
                    # Decide whether to continue or stop
                    if len(failed_builds) >= 2:
                        self.logger.error("❌ Multiple failures detected, stopping build process")
                        break
                
                # Wait between builds to let Docker settle
                if i < len(images_to_build):
                    self.logger.info("⏸️ Waiting 30 seconds between builds...")
                    import time
                    time.sleep(30)
            
            # Final cleanup
            self.docker_cleanup_between_builds()
            
            # Build summary
            self.logger.info("📊 Docker Build Summary:")
            self.logger.info(f"✅ Successful builds ({len(successful_builds)}): {successful_builds}")
            if failed_builds:
                self.logger.error(f"❌ Failed builds ({len(failed_builds)}): {failed_builds}")
            
            # Final space check
            self.check_disk_space("after all builds")
            
            if len(failed_builds) == 0:
                self.logger.info("🎉 All Docker images built and uploaded successfully!")
                return True
            else:
                self.logger.error(f"❌ {len(failed_builds)} image(s) failed to build")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build Docker images: {e}")
            self.docker_cleanup_between_builds()  # Cleanup on error
            return False

    def run_steps(self, steps_only=None):
        """Run step 6"""
        results = {}
        if not steps_only or 6 in steps_only:
            results[6] = self.step6_build_upload_docker()
        return results
