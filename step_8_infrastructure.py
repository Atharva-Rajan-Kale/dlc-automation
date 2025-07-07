import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional
from common import BaseAutomation, ECRImageSelector

class Step8InfrastructureAutomation(BaseAutomation):
    """Handles Step 8: Infrastructure deployment after PR merge"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.python_version_info = None
        self.infrastructure_dir = None
    
    def set_python_version_info(self, python_info: Dict):
        """Set python version info from external source (e.g., from steps 3-4)"""
        self.python_version_info = python_info
        self.logger.info(f"✅ Python version info set externally: {python_info}")
    
    def get_current_python_version_from_ecr(self) -> str:
        """Get current python version by querying ECR directly"""
        try:
            self.logger.info("🔍 Querying ECR to get current python version...")
            ecr_selector = ECRImageSelector()
            image_selection = ecr_selector.select_matching_cuda_images()
            sample_image = image_selection['training_cpu']
            self.logger.info(f"Extracting python version from image: {sample_image.tag}")
            python_match = re.search(r'-py(\d+)-', sample_image.tag)
            python_version = f"py{python_match.group(1)}"
            self.logger.info(f"✅ Extracted python version from ECR: {python_version}")
            return python_version
        except Exception as e:
            self.logger.warning(f"ECR query failed: {e}, defaulting to py311")
            return "py311"
    
    def get_python_version_enum(self) -> str:
        """Get python version enum format (e.g., py311 -> PY311)"""
        # First check if python version info was set externally
        if self.python_version_info and 'python_version' in self.python_version_info:
            python_version = self.python_version_info['python_version']
            self.logger.info(f"Using externally provided python version: {python_version}")
        else:
            # Fallback: Get current python version directly from ECR
            self.logger.info("No external python version provided, querying ECR...")
            python_version = self.get_current_python_version_from_ecr()
        
        # Convert py311 -> PY311
        python_enum = python_version.upper()
        self.logger.info(f"Using python version: {python_enum}")
        return python_enum
    
    def prompt_pr_merged(self) -> bool:
        """Prompt user if PR is merged"""
        while True:
            response = input(f"\n🔍 Has the AutoGluon {self.current_version} PR been merged? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                self.logger.info("✅ PR confirmed as merged, proceeding with infrastructure deployment")
                return True
            elif response in ['n', 'no']:
                self.logger.info("⏸️ PR not yet merged, stopping infrastructure deployment")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def prompt_cr_merged(self) -> bool:
        """Prompt user if CR is reviewed and merged"""
        while True:
            response = input(f"\n🔍 Has the infrastructure CR been reviewed and merged? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                self.logger.info("✅ CR confirmed as merged, infrastructure deployment complete")
                return True
            elif response in ['n', 'no']:
                self.logger.info("⏳ CR not yet merged, please wait for review and merge")
                print("💡 Please check your CR status and try again when it's merged")
                continue
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def setup_brazil_workspace(self) -> bool:
        """Setup Brazil workspace for infrastructure changes"""
        self.logger.info("Setting up Brazil workspace for DLContainersInfra")
        main_project_parent = self.main_project_dir.parent
        original_dir = os.getcwd()
        try:
            os.chdir(main_project_parent)
            self.logger.info(f"Changed to directory: {os.getcwd()}")
            
            workspace_dir = main_project_parent / "DLContainersInfra"
            
            # Check if workspace already exists
            if workspace_dir.exists():
                self.logger.info(f"📁 Workspace directory already exists: {workspace_dir}")
                self.logger.info("Skipping workspace creation, using existing directory...")
                os.chdir(workspace_dir)
                self.infrastructure_dir = workspace_dir
                self.logger.info(f"Changed to existing workspace directory: {os.getcwd()}")
            else:
                self.logger.info("Creating Brazil workspace...")
                subprocess.run(["brazil", "ws", "create", "-n", "DLContainersInfra"], check=True)
                os.chdir(workspace_dir)
                self.infrastructure_dir = workspace_dir
                self.logger.info(f"Changed to new workspace directory: {os.getcwd()}")
            
            # Setup workspace (always do this to ensure it's up to date)
            self.logger.info("Setting up workspace with mainline...")
            subprocess.run(["brazil", "ws", "use", "-vs", "DLContainersInfra/mainline"], check=True)
            self.logger.info("Adding DLContainersInfraCDK package...")
            subprocess.run(["brazil", "ws", "use", "-p", "DLContainersInfraCDK"], check=True)
            
            # Sync workspace to get latest changes
            self.logger.info("🔄 Syncing workspace to get latest changes...")
            subprocess.run(["brazil", "ws", "sync", "--md"], check=True)
            
            self.logger.info("Changing to CDK directory...")
            os.chdir("src/DLContainersInfraCDK")
            self.logger.info(f"Changed to CDK directory: {os.getcwd()}")
            config_path = Path("lib/config/public_release.ts")
            if not config_path.exists():
                self.logger.error(f"Config file not found: {config_path.absolute()}")
                self.logger.error("This suggests we're not in the correct directory structure")
                return False
            self.logger.info("✅ Brazil workspace setup completed")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Brazil command failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Workspace setup failed: {e}")
            return False
        finally:
            pass
    
    def update_public_release_config(self) -> bool:
        """Update lib/config/public_release.ts with new version"""
        config_path = Path("lib/config/public_release.ts")
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path.absolute()}")
            return False
        self.logger.info(f"Updating infrastructure config: {config_path}")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            original_content = content
            python_enum = self.get_python_version_enum()
            if self.is_major_release:
                success = self.update_config_for_major_release(content, python_enum)
            else:
                success = self.update_config_for_minor_release(content, python_enum)
            if success:
                with open(config_path, 'w') as f:
                    f.write(success)
                self.logger.info("✅ Infrastructure config updated successfully")
                return True
            else:
                self.logger.error("❌ Failed to update infrastructure config")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to update config file: {e}")
            return False
    
    def update_config_for_major_release(self, content: str, python_enum: str) -> Optional[str]:
        """Update config for major version release"""
        self.logger.info(f"Updating config for MAJOR release: {self.previous_version} -> {self.current_version}")
        curr_parts = self.current_version.split('.')
        curr_major = int(curr_parts[0])
        curr_minor = int(curr_parts[1])
        version_to_remove = f"{curr_major}.{curr_minor-2}.0"
        self.logger.info(f"Removing old version: {version_to_remove}")
        self.logger.info(f"Keeping previous version: {self.previous_version}")
        self.logger.info(f"Adding new version: {self.current_version}")
        for image_type in ["TRAINING", "INFERENCE"]:
            self.logger.info(f"Processing {image_type} configurations...")
            # Remove the old version block (e.g., 1.2.0)
            remove_pattern = rf'\s*{{\s*framework:\s*"autogluon",\s*version:\s*"{re.escape(version_to_remove)}",\s*image_type:\s*ImageType\.{image_type}[^}}]*}},?\s*'
            content = re.sub(remove_pattern, '', content, flags=re.DOTALL)
            self.logger.info(f"Removed {version_to_remove} block for {image_type}")
            # Find the previous version block and update it + add new block
            prev_block_pattern = rf'(\s*)({{\s*framework:\s*"autogluon",\s*version:\s*"{re.escape(self.previous_version)}",\s*image_type:\s*ImageType\.{image_type}[^}}]*python_versions:\s*\[[^\]]*)(PythonVersion\.PY\d+)([^\]]*\][^}}]*}})'
            def replace_and_add_new(match):
                indent = match.group(1)  # Capture the original indentation
                # Update python version in existing block
                updated_prev_block = f"{indent}{match.group(2)}PythonVersion.{python_enum}{match.group(4)}"
                
                # Add new version block with same indentation
                new_block = f''',
  {{
    framework: "autogluon",
    version: "{self.current_version}",
    image_type: ImageType.{image_type},
    python_versions: [PythonVersion.{python_enum}],
    device_types: [Device.CPU, Device.GPU],
    archType: ArchType.X86,
  }}'''
                return updated_prev_block + new_block
            content = re.sub(prev_block_pattern, replace_and_add_new, content, flags=re.DOTALL)
            self.logger.info(f"Updated {self.previous_version} and added {self.current_version} for {image_type}")
        content = re.sub(r'(// Autogluon Release Pipelines)\s*({)', r'\1\n  \2', content)
        autogluon_end_pattern = rf'(\s*{{\s*framework:\s*"autogluon"[^}}]*archType:\s*ArchType\.X86,\s*}})\s*(\s*//\s*[^{{\n]*)'
        content = re.sub(autogluon_end_pattern, r'\1,\n\2', content, flags=re.DOTALL)
        return content
    
    def update_config_for_minor_release(self, content: str, python_enum: str) -> Optional[str]:
        """Update config for minor version release"""
        self.logger.info(f"Updating config for MINOR release: {self.previous_version} -> {self.current_version}")
        for image_type in ["TRAINING", "INFERENCE"]:
            self.logger.info(f"Processing {image_type} configurations...")
            # Update version from previous to current (e.g., 1.3.0 -> 1.3.1)
            pattern = rf'({{[^}}]*framework:\s*"autogluon"[^}}]*version:\s*")({re.escape(self.previous_version)})("[^}}]*image_type:\s*ImageType\.{image_type}[^}}]*python_versions:\s*\[[^\]]*)(PythonVersion\.PY\d+)([^\]]*\][^}}]*}})'
            def replace_minor(match):
                return f"{match.group(1)}{self.current_version}{match.group(3)}PythonVersion.{python_enum}{match.group(5)}"
            content = re.sub(pattern, replace_minor, content, flags=re.DOTALL)
            self.logger.info(f"Updated {self.previous_version} to {self.current_version} for {image_type}")
        return content
    
    def commit_changes(self) -> bool:
        """Commit changes locally before submitting CR"""
        self.logger.info("Committing infrastructure changes...")
        try:
            subprocess.run(["git", "add", "."], check=True)
            commit_message = f"AutoGluon {self.current_version}: Update infrastructure deployment config"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            self.logger.info(f"✅ Changes committed locally with message: {commit_message}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Git commit failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Commit failed: {e}")
            return False
    
    def submit_cr(self) -> bool:
        """Commit changes and submit code review"""
        self.logger.info("Committing changes and submitting code review...")
        try:
            if not self.commit_changes():
                return False
            self.logger.info("Submitting code review...")
            result = subprocess.run(["cr"], check=True, capture_output=True, text=True)
            self.logger.info("✅ Code review submitted successfully")
            self.logger.info(f"CR output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ CR submission failed: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"❌ CR submission failed: {e}")
            return False
    
    def run_infrastructure_deployment(self) -> bool:
        """Run the complete infrastructure deployment step"""
        self.logger.info("🏗️ Starting infrastructure deployment step")
        try:
            if not self.prompt_pr_merged():
                return False
            if not self.setup_brazil_workspace():
                return False
            if not self.update_public_release_config():
                return False
            if not self.submit_cr():
                return False
            if not self.prompt_cr_merged():
                return False
            self.logger.info("✅ Infrastructure deployment completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Infrastructure deployment failed: {e}")
            return False
    
    def run_steps(self, steps_only=None):
        """Run step 8 (infrastructure deployment)"""
        results = {}
        if not steps_only or 8 in steps_only:
            results[8] = self.run_infrastructure_deployment()
        return results