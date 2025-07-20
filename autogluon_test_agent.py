import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional
import boto3
from common import BaseAutomation
from automation_logger import LoggerMixin
import uuid

class AutoGluonTestAgent(BaseAutomation, LoggerMixin):
    """Simple test runner for AutoGluon tests on the most recent training CPU image"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.test_results = {}
        self.test_files_dir = Path(__file__).parent / "autogluon_test_files"
        # Test configurations
        self.test_configs = [
            {
                "name": "tabular",
                "module": "test_tabular", 
                "function": "test_tabular"
            },
            {
                "name": "automm",
                "module": "test_automm",
                "function": "test_automm"
            },
            {
                "name": "ts",
                "module": "test_ts", 
                "function": "test_ts"
            }
        ]
        self.setup_logging(current_version,custom_name="autogluon_tests")

    def get_latest_training_cpu_image(self) -> Optional[str]:
        """Get the most recent training CPU image from beta-autogluon-training repository"""
        self.logger.info("🔍 Getting the most recent training CPU image...")
        account_id = os.environ.get('ACCOUNT_ID')
        region = os.environ.get('REGION', 'us-east-1')
        
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set")
            
        ecr_client = boto3.client('ecr', region_name=region)
        repo = 'beta-autogluon-training'
        
        try:
            response = ecr_client.describe_images(
                repositoryName=repo,
                maxResults=50
            )
            
            images = sorted(
                response['imageDetails'], 
                key=lambda x: x['imagePushedAt'], 
                reverse=True
            )
            
            self.logger.info(f"📦 Found {len(images)} total images in {repo}")
            
            # Debug: Show first few image tags
            for i, image in enumerate(images[:5]):
                if 'imageTags' in image:
                    tag = image['imageTags'][0]
                    self.logger.info(f"   Image {i+1}: {tag}")
            
            # Find the most recent CPU training image
            for image in images:
                if 'imageTags' in image:
                    tag = image['imageTags'][0]
                    # More flexible filtering - just look for CPU since we're already in training repo
                    if '-cpu-' in tag:
                        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
                        self.logger.info(f"📦 Found most recent CPU training image: {tag}")
                        return image_uri
            
            self.logger.warning("⚠️ No CPU training images found")
            self.logger.info("Available tags (first 10):")
            for i, image in enumerate(images[:10]):
                if 'imageTags' in image:
                    tag = image['imageTags'][0]
                    self.logger.info(f"   - {tag}")
            return None
                        
        except Exception as e:
            self.logger.error(f"❌ Failed to get images from {repo}: {e}")
            return None

    def validate_test_files(self) -> bool:
        """Validate that all required test files exist"""
        self.logger.info(f"🔍 Validating test files in {self.test_files_dir}")
        
        if not self.test_files_dir.exists():
            self.logger.error(f"❌ Test files directory not found: {self.test_files_dir}")
            return False
            
        required_files = ["test_automm.py", "test_tabular.py", "test_ts.py"]
        missing_files = []
        
        for test_file in required_files:
            file_path = self.test_files_dir / test_file
            if not file_path.exists():
                missing_files.append(test_file)
            else:
                self.logger.info(f"✅ Found: {test_file}")
                
        if missing_files:
            self.logger.error(f"❌ Missing test files: {missing_files}")
            return False
            
        return True

    def run_test_in_docker(self, image_uri: str, test_config: Dict) -> bool:
        """Run a specific test function in a Docker container"""
        test_name = test_config["name"]
        test_module = test_config["module"]
        test_function = test_config["function"]
        
        print(f"\n{'='*70}")
        print(f"🧪 RUNNING {test_name.upper()} TEST")
        print(f"{'='*70}")
        print(f"📦 Image: {image_uri.split('/')[-1]}")
        print(f"🐍 Module: {test_module}")
        print(f"⚙️  Function: {test_function}")
        
        container_name = f"autogluon-test-{test_name}-{uuid.uuid4().hex[:8]}"
        container_mount_path = "/autogluon_tests/"
        
        print(f"🔄 Ensuring image is available locally...")
        pull_cmd = ["docker", "pull", image_uri]
        try:
            result = subprocess.run(pull_cmd, capture_output=False, timeout=600)
            if result.returncode != 0:
                print(f"⚠️ Image pull had issues, but continuing...")
        except subprocess.TimeoutExpired:
            print(f"⚠️ Image pull timed out after 10 minutes, but continuing...")
        except Exception as e:
            print(f"⚠️ Image pull error: {e}, but continuing...")
        
        create_cmd = [
            "docker", "create", "--name", container_name,
            "--shm-size=2g",
            "-w", container_mount_path,
            image_uri,
            "sleep", "3600"
        ]
        
        print(f"🐳 Creating container: {container_name}")
        
        try:
            result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=300) 
            if result.returncode != 0:
                print(f"❌ Failed to create container: {result.stderr}")
                return False
            
            print(f"📁 Copying test files to container...")
            copy_cmd = [
                "docker", "cp", 
                str(self.test_files_dir) + "/.",  # Copy contents of directory
                f"{container_name}:{container_mount_path}"
            ]
            
            result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"❌ Failed to copy files: {result.stderr}")
                return False
            start_cmd = ["docker", "start", container_name]
            result = subprocess.run(start_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"❌ Failed to start container: {result.stderr}")
                return False
            
            test_command = f"cd {container_mount_path} && python -c 'from {test_module} import {test_function}; {test_function}()'"
            
            exec_cmd = [
                "docker", "exec", container_name,
                "bash", "-c", test_command
            ]
            
            print(f"🚀 Executing test in container...")
            print(f"   Command: {test_command}")
            print("-" * 70)
            
            try:
                result = self.run_subprocess_with_logging(
                    exec_cmd,
                    capture_output=False,
                    timeout=1800 
                )
            except TypeError:
                result = subprocess.run(
                    exec_cmd,
                    capture_output=False,
                    timeout=1800
                )
            
            success = result.returncode == 0
            
            print("-" * 70)
            if success:
                print(f"✅ {test_name.upper()} TEST PASSED")
            else:
                print(f"❌ {test_name.upper()} TEST FAILED (Exit code: {result.returncode})")
            print(f"{'='*70}\n")
                
            return success
            
        except subprocess.TimeoutExpired as e:
            print("-" * 70)
            print(f"⏰ {test_name.upper()} TEST TIMED OUT: {str(e)}")
            print(f"{'='*70}\n")
            return False
            
        except KeyboardInterrupt:
            print("-" * 70)
            print(f"🛑 {test_name.upper()} TEST INTERRUPTED BY USER")
            print(f"{'='*70}\n")
            return False
            
        except Exception as e:
            print("-" * 70)
            print(f"❌ {test_name.upper()} TEST EXECUTION ERROR: {str(e)}")
            print(f"{'='*70}\n")
            return False
            
        finally:
            try:
                cleanup_cmd = ["docker", "rm", "-f", container_name]
                subprocess.run(cleanup_cmd, capture_output=True, timeout=30)
                print(f"🧹 Cleaned up container: {container_name}")
            except:
                pass

    def run_autogluon_test_agent(self) -> bool:
        """Main execution loop for AutoGluon tests"""
        self.logger.info("🤖 Starting AutoGluon Test Agent...")
        
        try:
            if not self.validate_test_files():
                return False
            
            # Get the most recent training CPU image
            image_uri = self.get_latest_training_cpu_image()
            if not image_uri:
                self.logger.error("❌ No training CPU image found")
                return False
            
            self.logger.info(f"🎯 Testing image: {image_uri.split('/')[-1]}")
            overall_success = True
            
            # Run all tests
            for test_config in self.test_configs:
                test_name = test_config["name"]
                success = self.run_test_in_docker(image_uri, test_config)
                self.test_results[test_name] = success
                
                if not success:
                    overall_success = False
            
            self.print_test_summary()
            return overall_success
            
        except Exception as e:
            self.logger.error(f"❌ AutoGluon Test Agent failed: {e}")
            return False

    def print_test_summary(self):
        """Print test execution summary"""
        print("\n" + "="*70)
        print("🧪 AUTOGLUON TEST AGENT SUMMARY")
        print("="*70)
        
        passed = [name for name, passed in self.test_results.items() if passed]
        failed = [name for name, passed in self.test_results.items() if not passed]
        
        print(f"📊 Total Tests: {len(self.test_results)} | ✅ Passed: {len(passed)} | ❌ Failed: {len(failed)}")
        
        if passed:
            print("\n✅ Passed tests:")
            for test_name in passed:
                print(f"   - {test_name}")
        
        if failed:
            print("\n❌ Failed tests:")
            for test_name in failed:
                print(f"   - {test_name}")
        
        print("="*70)

def main():
    """Main function for AutoGluon Test Agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoGluon Test Agent')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.1)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.0)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    
    args = parser.parse_args()
    
    agent = AutoGluonTestAgent(
        args.current_version,
        args.previous_version, 
        args.fork_url
    )
    
    success = agent.run_autogluon_test_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()