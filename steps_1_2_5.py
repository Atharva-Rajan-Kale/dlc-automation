import os
import re
import shutil
import subprocess
from pathlib import Path
from common import BaseAutomation

class Steps125Automation(BaseAutomation):
    """Handles Steps 1, 2, and 5:Branch creation, TOML update, and Package model"""
    
    def step1_create_branch(self):
        """Step 1:Cut a new branch in fork to work on a new release"""
        self.logger.info("Step 1:Creating release branch")
        self.workspace_dir.mkdir(exist_ok=True)
        original_dir=os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            if not Path("deep-learning-containers").exists():
                self.logger.info(f"Cloning from {self.fork_url}")
                subprocess.run(["git", "clone", self.fork_url, "deep-learning-containers"], check=True)
            os.chdir("deep-learning-containers")
            result=subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
            self.logger.info(f"Working in repository:{result.stdout.strip()}")
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
            branch_name=f"autogluon-{self.current_version}-release"
            self.logger.info(f"Creating branch:{branch_name}")
            try:
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            except:
                subprocess.run(["git", "checkout", branch_name], check=True)
            self.logger.info("✅ Step 1 completed:Release branch created")
            return True
        except Exception as e:
            os.chdir(original_dir)
            self.logger.error(f"❌ Step 1 failed:{e}")
            return False

    def step2_update_toml(self):
        """Step 2:Update toml file to build only AutoGluon"""
        self.logger.info("Step 2:Updating TOML configuration")
        original_dir=os.getcwd()
        try:
            if not self.repo_dir.exists():
                self.logger.error(f"Repository directory not found:{self.repo_dir}")
                return False
            
            os.chdir(self.repo_dir)
            toml_path=Path("dlc_developer_config.toml")
            if not toml_path.exists():
                self.logger.error(f"TOML file not found:{toml_path.absolute()}")
                return False
            with open(toml_path, 'r') as f:
                content=f.read()
            content=re.sub(
                r'build_frameworks\s*=\s*\[.*?\]',
                'build_frameworks=["autogluon"]',
                content,
                flags=re.DOTALL
            )
            self.logger.info("Updated build_frameworks=['autogluon']")
            content=re.sub(
                r'dlc-pr-autogluon-training\s*=\s*""',
                'dlc-pr-autogluon-training="autogluon/training/buildspec.yml"',
                content
            )
            self.logger.info("Updated dlc-pr-autogluon-training buildspec path")
            content=re.sub(
                r'dlc-pr-autogluon-inference\s*=\s*""',
                'dlc-pr-autogluon-inference="autogluon/inference/buildspec.yml"',
                content
            )
            self.logger.info("Updated dlc-pr-autogluon-inference buildspec path")
            with open(toml_path, 'w') as f:
                f.write(content)
            subprocess.run(["git", "add", str(toml_path)], check=True)
            subprocess.run(["git", "commit", "-m", 
                           f"AutoGluon {self.current_version}:Update TOML for AutoGluon-only build"], 
                          check=True)
            self.logger.info("✅ Step 2 completed:TOML updated and committed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Step 2 failed:{e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def step5_package_model(self):
        """Step 5:Update package_model.py version and execute it"""
        self.logger.info("Step 5:Packaging model")
        package_model_path=self.main_project_dir / "package_model.py"
        self.logger.info(f"Main project dir:{self.main_project_dir}")
        self.logger.info(f"Looking for package_model.py in:{self.main_project_dir}")
        if not package_model_path.exists():
            self.logger.error(f"package_model.py not found at:{package_model_path}")
            return False
        original_cwd=os.getcwd()
        try:
            with open(package_model_path, 'r') as f:
                content=f.read()
            content=re.sub(r"version\s*=\s*['\"][\d.]+['\"]", f"version='{self.current_version}'", content)
            with open(package_model_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Updated version to {self.current_version}")
            os.chdir(self.main_project_dir)
            self.logger.info(f"Changed to:{os.getcwd()}")
            self.logger.info("Executing package_model.py...")
            result=subprocess.run(['python', 'package_model.py'], check=True)
            self.logger.info("✅ Model training completed")
            source_file=self.main_project_dir / f"model_{self.current_version}.tar.gz"
            target_dir=self.repo_dir / "test/sagemaker_tests/autogluon/inference/resources/model"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file=target_dir / f"model_{self.current_version}.tar.gz"
            shutil.move(str(source_file), str(target_file))
            self.logger.info(f"✅ Moved model to:{target_file}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Step 5 failed:{e}")
            return False
        finally:
            os.chdir(original_cwd)
            self.logger.info(f"Returned to:{os.getcwd()}")

    def run_steps(self, steps_only=None):
        """Run steps 1, 2, and 5"""
        results={}
        if not steps_only or 1 in steps_only:
            results[1]=self.step1_create_branch()
        if not steps_only or 2 in steps_only:
            results[2]=self.step2_update_toml()
        if not steps_only or 5 in steps_only:
            results[5]=self.step5_package_model()
        return results
