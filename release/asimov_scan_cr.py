
import os
import re
import sys
import subprocess
import logging
import boto3
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError
from automation.automation_logger import LoggerMixin
class AsimovSecurityScanAutomation(LoggerMixin):
    """Automation for AsimovImageSecurityScan workspace and images.py updates"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        self.current_version = current_version
        self.previous_version = previous_version
        self.fork_url = fork_url
        # Determine if this is a major version update
        current_parts = current_version.split('.')
        previous_parts = previous_version.split('.')
        self.is_major_release = (
            current_parts[0] != previous_parts[0] or 
            (current_parts[1] != previous_parts[1] and current_parts[2] == '0')
        )
        # Get major version numbers for tagging
        self.current_major_minor = f"{current_parts[0]}.{current_parts[1]}"
        self.previous_major_minor = f"{previous_parts[0]}.{previous_parts[1]}"        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AsimovImageSecurityScan Automation - Version {current_version}")
        self.logger.info(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
        # Workspace paths
        self.script_dir = Path(__file__).parent.parent.parent
        self.workspace_dir = self.script_dir / "AsimovImageSecurityScan"
        self.images_py_path = self.workspace_dir / "src" / "AsimovImageSecurityScan" / "src" / "asimov_image_security_scan" / "images.py"
        self.setup_logging(current_version,custom_name="asimov_scan")
    def setup_brazil_workspace(self) -> bool:
        """Set up the Brazil workspace"""
        try:
            self.logger.info("🏗️ Setting up Brazil workspace...")
            # Change to script directory
            original_dir = os.getcwd()
            os.chdir(self.script_dir)
            # Check if workspace already exists
            if self.workspace_dir.exists():
                self.logger.info(f"📁 Workspace directory already exists: {self.workspace_dir}")
                self.logger.info("Skipping workspace creation, using existing directory...")
                os.chdir(self.workspace_dir)
            else:
                # Create workspace
                self.logger.info("📁 Creating Brazil workspace...")
                result = self.run_subprocess_with_logging(
                    ["brazil", "ws", "create", "-n", "AsimovImageSecurityScan"],
                    "Creating Brazil workspace",
                    capture_output=False
                )
                if result.returncode != 0:
                    self.logger.error(f"❌ Failed to create workspace")
                    return False
                # Change to workspace directory
                os.chdir(self.workspace_dir)
            # Use version set
            self.logger.info("📦 Setting up version set...")
            result = self.run_subprocess_with_logging(
                ["brazil", "ws", "use", "-vs", "AsimovPassiveScripts/mainline"],
                "Setting up version set",
                capture_output=False
            )
            if result.returncode != 0:
                self.logger.error(f"❌ Failed to use version set")
                return False
            # Use package
            self.logger.info("📦 Setting up package...")
            result = self.run_subprocess_with_logging(
                ["brazil", "ws", "use", "-p", "AsimovImageSecurityScan"],
                "Setting up package",
                capture_output=False
            )
            if result.returncode != 0:
                self.logger.error(f"❌ Failed to use package")
                return False
            # Sync workspace
            self.logger.info("🔄 Syncing workspace to get latest changes...")
            result = self.run_subprocess_with_logging(
                ["brazil", "ws", "sync", "--md"],
                "Syncing workspace",
                capture_output=False
            )
            if result.returncode != 0:
                self.logger.error(f"❌ Failed to sync workspace")
                return False
            # Change to the actual package directory
            package_dir = self.workspace_dir / "src" / "AsimovImageSecurityScan"
            if package_dir.exists():
                os.chdir(package_dir)
                self.logger.info(f"📁 Changed to package directory: {package_dir}")
            else:
                self.logger.warning(f"⚠️ Package directory not found: {package_dir}")
            self.logger.info("✅ Brazil workspace setup complete")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error setting up Brazil workspace: {e}")
            return False
            
    def get_latest_autogluon_image_info(self) -> Optional[Dict]:
        """Get the absolute latest AutoGluon training image to extract python version"""
        try:
            self.logger.info("🔍 Getting absolute latest AutoGluon training image...")
            account_id = '763104351884'
            region = 'us-west-2'
            ecr_client = boto3.client('ecr', region_name=region)
            try:
                response = ecr_client.describe_images(
                    registryId=account_id,
                    repositoryName='autogluon-training',
                    maxResults=50
                )
                images = sorted(
                    response['imageDetails'], 
                    key=lambda x: x['imagePushedAt'], 
                    reverse=True
                )
                self.logger.info(f"📦 Found {len(images)} images in autogluon-training")
                for image in images:
                    if 'imageTags' in image and image['imageTags']:
                        tag = image['imageTags'][0]
                        self.logger.info(f"📦 Absolute latest image: {tag}")
                        python_match = re.search(r'-py(\d+)-', tag)
                        if python_match:
                            python_version = f"py{python_match.group(1)}"
                            if python_version in ['py312']:
                                python_path = "/usr/local/bin/python"
                            else:
                                python_path = "/opt/conda/bin/python"
                            self.logger.info(f"🐍 Python version: {python_version} -> {python_path}")
                            return {
                                'python_version': python_version,
                                'python_path': python_path,
                                'tag': tag
                            }
                        else:
                            self.logger.warning(f"⚠️ Could not parse python version from tag: {tag}")
                            continue
                self.logger.warning("⚠️ No images with valid python version found")
                return self._get_default_python_info()
            except ClientError as e:
                self.logger.error(f"❌ ECR describe_images failed: {e}")
                return self._get_default_python_info()
        except Exception as e:
            self.logger.error(f"❌ Failed to get image info: {e}")
            return self._get_default_python_info()
    
    def _get_default_python_info(self) -> Dict:
        """Return default python info when ECR access fails"""
        self.logger.info("📦 Using default python configuration")
        return {
            'python_version': 'py311',
            'python_path': '/opt/conda/bin/python',
            'tag': 'default'
        }
            
    def update_images_py(self, python_info: Dict) -> bool:
        """Update the images.py file with new AutoGluon version entries"""
        try:
            if not self.is_major_release:
                self.logger.info("ℹ️ Minor version update - no changes needed to images.py")
                return True
            self.logger.info("📝 Updating images.py for major version update...")
            images_py_relative_path = Path("src") / "asimov_image_security_scan" / "images.py"
            if not images_py_relative_path.exists():
                self.logger.error(f"❌ images.py not found at: {images_py_relative_path.absolute()}")
                return False
            with open(images_py_relative_path, 'r') as f:
                content = f.read()
            updated_content = self.update_autogluon_entries(content, python_info)
            with open(images_py_relative_path, 'w') as f:
                f.write(updated_content)
            self.logger.info("✅ images.py updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update images.py: {e}")
            return False
            
    def update_autogluon_entries(self, content: str, python_info: Dict) -> str:
        """Update AutoGluon entries in the file content"""
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if ('"763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference"' in line or
                '"763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training"' in line):
                repo_type = "inference" if "inference" in line else "training"
                self.logger.info(f"📝 Updating autogluon-{repo_type} section")
                new_lines.append(line)
                i += 1
                if i < len(lines) and '[' in lines[i]:
                    new_lines.append(lines[i])
                    i += 1
                existing_entries = self.extract_existing_entries(lines, i)
                new_entries = self.generate_autogluon_entries_with_shift(existing_entries, python_info)
                new_lines.extend(new_entries)
                while i < len(lines) and '],' not in lines[i]:
                    i += 1
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            else:
                new_lines.append(line)
                i += 1
        return '\n'.join(new_lines)
    
    def extract_existing_entries(self, lines: List[str], start_index: int) -> List[Dict]:
        """Extract existing entries from the file"""
        entries = []
        i = start_index
        current_entry = {}
        while i < len(lines) and '],' not in lines[i]:
            line = lines[i].strip()
            if '"python":' in line:
                python_match = re.search(r'"python":\s*"([^"]+)"', line)
                if python_match:
                    current_entry['python'] = python_match.group(1)
            elif '"tag":' in line:
                tag_match = re.search(r'"tag":\s*"([^"]+)"', line)
                if tag_match:
                    current_entry['tag'] = tag_match.group(1)
            elif '},' in line and current_entry:
                entries.append(current_entry.copy())
                current_entry = {}
            i += 1
        return entries
        
    def generate_autogluon_entries_with_shift(self, existing_entries: List[Dict], python_info: Dict) -> List[str]:
        """Generate new AutoGluon entries with proper version shifting"""
        entries = []
        entries.extend([
            '            {',
            f'                "python": "{python_info["python_path"]}",',
            f'                "tag": "{self.current_major_minor}-cpu-{python_info["python_version"]}",',
            '            },',
            '            {',
            f'                "python": "{python_info["python_path"]}",',
            f'                "tag": "{self.current_major_minor}-gpu-{python_info["python_version"]}",',
            '            },'
        ])
        if len(existing_entries) >= 2:
            first_entry = existing_entries[0]
            second_entry = existing_entries[1]
            first_tag = first_entry['tag'].replace(self.previous_major_minor, self.previous_major_minor)
            second_tag = second_entry['tag'].replace(self.previous_major_minor, self.previous_major_minor)
            entries.extend([
                '            {',
                f'                "python": "{first_entry["python"]}",',
                f'                "tag": "{first_tag}",',
                '            },',
                '            {',
                f'                "python": "{second_entry["python"]}",',
                f'                "tag": "{second_tag}",',
                '            }'
            ])
        else:
            entries.extend([
                '            {',
                f'                "python": "{python_info["python_path"]}",',
                f'                "tag": "{self.previous_major_minor}-cpu-{python_info["python_version"]}",',
                '            },',
                '            {',
                f'                "python": "{python_info["python_path"]}",',
                f'                "tag": "{self.previous_major_minor}-gpu-{python_info["python_version"]}",',
                '            }' 
            ])
        return entries
        
    def send_cr(self) -> bool:
        """Send a CR (Code Review) using Brazil"""
        try:
            if not self.is_major_release:
                self.logger.info("ℹ️ Minor version update - no CR needed")
                return True
            self.logger.info("📤 Preparing CR...")
            current_dir = os.getcwd()
            self.logger.info(f"📁 Current directory: {current_dir}")
            self.logger.info("💾 Committing changes...")
            result = self.run_subprocess_with_logging(
                ["git", "add", "."],
                "Staging changes for commit",
                capture_output=False
            )
            if result.returncode != 0:
                self.logger.error(f"❌ Failed to stage changes")
                return False
            commit_message = f"AutoGluon {self.current_version}: Update security scan images\n\nAdding AutoGluon {self.current_version} images for security scanning.\nShifting previous version {self.previous_version} to older position."
            result = self.run_subprocess_with_logging(
                ["git", "commit", "-m", commit_message],
                "Committing changes",
                capture_output=False
            )
            if result.returncode != 0:
                self.logger.error(f"❌ Failed to commit changes")
                return False
            self.logger.info("✅ Changes committed successfully")
            self.logger.info("📤 Sending CR...")
            result = self.run_subprocess_with_logging(
                ["cr"],
                "Sending Code Review",
                capture_output=False
            )
            if result.returncode != 0:
                self.logger.error(f"❌ Failed to send CR")
                return False
            self.logger.info("✅ CR sent successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error sending CR: {e}")
            return False
            
    def run_automation(self) -> bool:
        """Run the complete automation workflow"""
        try:
            print("🚀 Starting AsimovImageSecurityScan Automation...")
            print(f"📋 Version: {self.previous_version} -> {self.current_version}")
            print(f"📋 Type: {'Major' if self.is_major_release else 'Minor'} release")
            print(f"📝 Subprocess logs: {self.automation_logger.get_log_file_path()}")
            # Setup Brazil workspace 
            if not self.setup_brazil_workspace():
                return False
            # Get python version info from latest image
            python_info = self.get_latest_autogluon_image_info()
            if not python_info:
                return False
            # Update images.py
            if not self.update_images_py(python_info):
                return False
            # Send CR if major release
            if not self.send_cr():
                return False
            print("✅ AsimovImageSecurityScan automation completed successfully!")
            if self.is_major_release:
                print("📋 Summary:")
                print(f"   - Workspace created/synced: {self.workspace_dir}")
                print(f"   - Package directory: {self.workspace_dir / 'src' / 'AsimovImageSecurityScan'}")
                print(f"   - images.py updated with {self.current_version} entries")
                print(f"   - Previous version {self.previous_version} shifted to older position")
                print(f"   - Changes committed and CR sent for review")
            else:
                print("📋 Summary:")
                print(f"   - Minor version update - no changes needed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Automation failed: {e}")
            return False

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='AsimovImageSecurityScan Automation')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.4.0)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.0)')
    parser.add_argument('--fork-url', required=True, help='Fork URL for the repository')
    args = parser.parse_args()
    automation = AsimovSecurityScanAutomation(
        args.current_version,
        args.previous_version,
        args.fork_url
    )
    success = automation.run_automation()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()