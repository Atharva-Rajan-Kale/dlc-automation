"""
title : Repository Setup and Model Packaging Automation

description : Manages initial repository setup, configuration updates, and model packaging for AutoGluon release automation. 
Step 1 creates release branches and syncs with upstream,
Step 2 updates TOML configuration for AutoGluon-only builds, and Step 5 trains and
packages a test model. Handles git operations, branch management, configuration file updates, and model training. 
Critical foundation steps that prepare the repository and resources needed for
subsequent Docker building and testing phases of the release process.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from automation.common import BaseAutomation
from automation.automation_logger import LoggerMixin

class Steps125Automation(BaseAutomation, LoggerMixin):
    """Handles Steps 1, 2, and 5: Branch creation, TOML update, and Package model"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_logging(current_version, custom_name="steps_1_2_5")

    def setup_git_config(self):
        """Setup git configuration for CI environment"""
        try:
            # Check if git user is already configured
            result = self.run_subprocess_with_logging(
                ["git", "config", "user.name"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if not result.stdout.strip():
                self.logger.info("Setting up git configuration for CI")
                self.run_subprocess_with_logging([
                    "git", "config", "user.name", "Atharva-Rajan-Kale"
                ], check=True)
                self.run_subprocess_with_logging([
                    "git", "config", "user.email", "atharvakale912@gmail.com"
                ], check=True)
                self.logger.info("✅ Git configuration set")
            else:
                self.logger.info(f"Git user already configured: {result.stdout.strip()}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to setup git config: {e}")
            return False

    def check_git_changes(self):
        """Check if there are any changes to commit"""
        try:
            # Check for unstaged changes
            result = self.run_subprocess_with_logging(
                ["git", "diff", "--quiet"], 
                capture_output=True, 
                check=False
            )
            has_unstaged = result.returncode != 0
            # Check for staged changes
            result = self.run_subprocess_with_logging(
                ["git", "diff", "--cached", "--quiet"], 
                capture_output=True, 
                check=False
            )
            has_staged = result.returncode != 0
            return has_unstaged or has_staged
        except Exception as e:
            self.logger.error(f"Error checking git changes: {e}")
            return False

    def step1_create_branch(self):
        """Step 1: Cut a new branch in fork to work on a new release"""
        self.logger.info("Step 1: Creating release branch")
        self.workspace_dir.mkdir(exist_ok=True)
        original_dir = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            if not Path("deep-learning-containers").exists():
                self.logger.info(f"Cloning from {self.fork_url}")
                self.run_subprocess_with_logging(["git", "clone", self.fork_url, "deep-learning-containers"], check=True)
            os.chdir("deep-learning-containers")
            # Setup git config early
            if not self.setup_git_config():
                return False
            result = self.run_subprocess_with_logging(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
            self.logger.info(f"Working in repository: {result.stdout.strip()}")
            try:
                self.run_subprocess_with_logging(["git", "remote", "get-url", "upstream"], capture_output=True, check=True)
                self.logger.info("Upstream remote already exists")
            except:
                self.logger.info("Adding upstream remote")
                self.run_subprocess_with_logging(["git", "remote", "add", "upstream", 
                              "https://github.com/aws/deep-learning-containers.git"], check=True)
            self.logger.info("Syncing with upstream...")
            self.run_subprocess_with_logging(["git", "fetch", "upstream"], check=True)
            self.run_subprocess_with_logging(["git", "checkout", "master"], check=True)
            self.run_subprocess_with_logging(["git", "reset", "--hard", "upstream/master"], check=True)
            
            branch_name = f"autogluon-{self.current_version}-release"
            self.logger.info(f"Creating branch: {branch_name}")
            try:
                self.run_subprocess_with_logging(["git", "checkout", "-b", branch_name], check=True)
            except:
                self.run_subprocess_with_logging(["git", "checkout", branch_name], check=True)
            
            self.logger.info("✅ Step 1 completed: Release branch created")
            return True
        except Exception as e:
            self.logger.error(f"❌ Step 1 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def step2_update_toml(self):
        """Step 2: Update toml file to build only AutoGluon"""
        self.logger.info("Step 2: Updating TOML configuration")
        original_dir = os.getcwd()
        try:
            if not self.repo_dir.exists():
                self.logger.error(f"Repository directory not found: {self.repo_dir}")
                return False
            
            os.chdir(self.repo_dir)
            
            # Ensure git config is set
            if not self.setup_git_config():
                return False
            
            toml_path = Path("dlc_developer_config.toml")
            if not toml_path.exists():
                self.logger.error(f"TOML file not found: {toml_path.absolute()}")
                return False
            
            # Read and update TOML content
            with open(toml_path, 'r') as f:
                original_content = f.read()
            
            content = original_content
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
            
            # Only write and commit if content actually changed
            if content != original_content:
                with open(toml_path, 'w') as f:
                    f.write(content)
                
                self.run_subprocess_with_logging(["git", "add", str(toml_path)], check=True)
                
                # Check if there are actually changes to commit
                if self.check_git_changes():
                    self.run_subprocess_with_logging(["git", "commit", "-m", 
                               f"AutoGluon {self.current_version}: Update TOML for AutoGluon-only build"], 
                              check=True)
                    self.logger.info("✅ Step 2 completed: TOML updated and committed")
                else:
                    self.logger.info("✅ Step 2 completed: TOML updated but no changes to commit")
            else:
                self.logger.info("✅ Step 2 completed: TOML file already up to date")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Step 2 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def step5_package_model(self):
        """Step 5: Update package_model.py version and execute it"""
        self.logger.info("Step 5: Packaging model")
        package_model_path = self.main_project_dir/"updation"/ "package_model.py"
        self.logger.info(f"Main project dir: {self.main_project_dir}")
        self.logger.info(f"Looking for package_model.py in: {self.main_project_dir}")
        
        if not package_model_path.exists():
            self.logger.error(f"package_model.py not found at: {package_model_path}")
            return False
        
        original_cwd = os.getcwd()
        try:
            # Update version in package_model.py
            with open(package_model_path, 'r') as f:
                content = f.read()
            
            content = re.sub(r"version\s*=\s*['\"][\d.]+['\"]", f"version='{self.current_version}'", content)
            
            with open(package_model_path, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Updated version to {self.current_version}")
            
            # Change to main project directory
            os.chdir(self.main_project_dir)
            self.logger.info(f"Changed to: {os.getcwd()}")
            
            # Debug environment before running package_model.py
            self.logger.info("🔍 Environment debug:")
            self.logger.info(f"Python version: {subprocess.run(['python', '--version'], capture_output=True, text=True).stdout.strip()}")
            self.logger.info(f"Current directory contents: {list(Path('.').iterdir())}")
            
            # Check if we have internet connectivity
            try:
                import requests
                response = requests.get('https://autogluon.s3.amazonaws.com', timeout=10)
                self.logger.info("✅ Internet connectivity confirmed")
            except Exception as e:
                self.logger.warning(f"⚠️ Internet connectivity issue: {e}")
            
            # Run package_model.py with timeout and better error handling
            self.logger.info("Executing package_model.py...")
            try:
                result = self.run_subprocess_with_logging(
                    ['python', 'updation/package_model.py'], 
                    check=True,
                    timeout=1800  # 30 minutes timeout
                )
                self.logger.info("✅ Model training completed")
            except subprocess.TimeoutExpired:
                self.logger.error("❌ package_model.py timed out after 30 minutes")
                return False
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ package_model.py failed with exit code {e.returncode}")
                if e.stdout:
                    self.logger.error(f"STDOUT: {e.stdout}")
                if e.stderr:
                    self.logger.error(f"STDERR: {e.stderr}")
                return False
            
            # Move the generated model file
            source_file = self.main_project_dir / f"model_{self.current_version}.tar.gz"
            if not source_file.exists():
                self.logger.error(f"❌ Expected model file not found: {source_file}")
                return False
            
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

    def run_steps(self, steps_only=None):
        """Run steps 1, 2, and 5"""
        self.logger.info("🔧 Running steps 1, 2, and 5...")
        results = {}
        
        if not steps_only or 1 in steps_only:
            results[1] = self.step1_create_branch()
        
        if not steps_only or 2 in steps_only:
            results[2] = self.step2_update_toml()
        
        if not steps_only or 5 in steps_only:
            results[5] = self.step5_package_model()
        
        return results
