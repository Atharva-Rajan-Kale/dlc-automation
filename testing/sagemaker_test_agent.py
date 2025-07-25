"""
title : SageMaker Test Automation Agent

description : Testing system that automatically runs SageMaker tests.
Executes local and remote SageMaker tests with proper IAM setup, analyzes
and prints out the failures. Handles complex test environments and manages 
ECR authentication. Features container-specific test routing, timeout management, 
and detailed reporting of test results across CPU/GPU variants for both training 
and inference workloads.
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import boto3

from automation.common import BaseAutomation, ECRImageSelector
from automation.automation_logger import LoggerMixin

class SageMakerTestAgent(BaseAutomation,LoggerMixin):
    """Test runner for SageMaker tests"""
    
    def __init__(self, current_version:str, previous_version:str, fork_url:str):
        super().__init__(current_version, previous_version, fork_url)
        self.test_results={}
        self.setup_logging(current_version,custom_name="sagemaker")

    def get_latest_ecr_images(self) -> Dict[str, List[str]]:
        """Get latest CPU and GPU images from the repository """
        account_id=os.environ.get('ACCOUNT_ID')
        region=os.environ.get('REGION', 'us-east-1')
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set")
        ecr_client=boto3.client('ecr', region_name=region)
        
        repo = 'beta-autogluon-inference'
        latest_images = {}
        
        try:
            response=ecr_client.describe_images(repositoryName=repo, maxResults=100)
            images=sorted(response['imageDetails'], key=lambda x:x['imagePushedAt'], reverse=True)
            
            latest_images[repo] = []
            cpu_found = False
            gpu_found = False
            for image in images:
                if 'imageTags' in image:
                    tag = image['imageTags'][0]
                    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
                    # Get latest CPU image
                    if '-cpu-' in tag and not cpu_found:
                        latest_images[repo].append(image_uri)
                        cpu_found = True
                        self.logger.info(f"📦 {repo} CPU: {tag}")
                    # Get latest GPU image
                    elif '-gpu-' in tag and not gpu_found:
                        latest_images[repo].append(image_uri)
                        gpu_found = True
                        self.logger.info(f"📦 {repo} GPU: {tag}")
                    # Break if we found both CPU and GPU images
                    if cpu_found and gpu_found:
                        break
            if not latest_images[repo]:
                self.logger.warning(f"⚠️ No CPU or GPU images found in {repo}")
        except Exception as e:
            self.logger.error(f"❌ Failed to get images from {repo}: {e}")
            latest_images[repo] = []
        
        return latest_images

    def setup_iam_permissions(self) -> bool:
        """Setup required IAM permissions for SageMaker tests"""
        try:
            result=self.run_subprocess_with_logging([
                "curl", "-s", "--connect-timeout", "5",
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0 or not result.stdout.strip():
                return True  
            instance_profile=result.stdout.strip()
            result=self.run_subprocess_with_logging([
                "aws", "iam", "get-instance-profile",
                "--instance-profile-name", instance_profile,
                "--query", "InstanceProfile.Roles[0].RoleName",
                "--output", "text"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return True
            role_name=result.stdout.strip()
            policy_document={
                "Version":"2012-10-17",
                "Statement":[{
                    "Effect":"Allow",
                    "Action":["ec2:*", "sagemaker:*", "iam:PassRole", "logs:*", "s3:*"],
                    "Resource":"*"
                }]
            }
            result=self.run_subprocess_with_logging([
                "aws", "iam", "get-role-policy",
                "--role-name", role_name,
                "--policy-name", "SageMakerTestingPolicy"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                self.run_subprocess_with_logging([
                    "aws", "iam", "put-role-policy",
                    "--role-name", role_name,
                    "--policy-name", "SageMakerTestingPolicy",
                    "--policy-document", json.dumps(policy_document)
                ], check=True)
                self.logger.info("✅ Created SageMaker testing policy")
            if not os.environ.get('SAGEMAKER_ROLE_ARN'):
                account_result=self.run_subprocess_with_logging([
                    "aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"
                ], capture_output=True, text=True, check=True)
                account_id=account_result.stdout.strip()
                os.environ['SAGEMAKER_ROLE_ARN']=f"arn:aws:iam::{account_id}:role/{role_name}"
                self.logger.info("✅ Set SAGEMAKER_ROLE_ARN")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ IAM setup failed:{e}")
            return True

    def setup_test_environment(self) -> bool:
        """Setup test environment for inference container"""
        test_dir=self.repo_dir / f"test/sagemaker_tests/autogluon/inference"
        try:
            self.setup_iam_permissions()
            os.environ['PYTHONPATH']=str(self.repo_dir / "src")
            account_id=os.environ.get('ACCOUNT_ID')
            region=os.environ.get('REGION', 'us-east-1')
            login_cmd=f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
            self.run_subprocess_with_logging(login_cmd, shell=True, check=True)
            os.chdir(test_dir)
            if Path("requirements.txt").exists():
                self.run_subprocess_with_logging(["pip3", "install", "-r", "requirements.txt"], check=True)
            self.logger.info(f"✅ Setup complete for inference")
            return True
        except Exception as e:
            self.logger.error(f"❌ Setup failed:{e}")
            return False

    def run_single_test_suite(self, image_uri: str, container_type: str, test_type: str) -> Tuple[bool, str]:
        """Run a single test suite (local or sagemaker) for a specific image"""
        tag = image_uri.split(':')[-1]
        processor = 'cpu' if '-cpu-' in tag else 'gpu'
        py_match = re.search(r'-py(\d+)-', tag)
        python_version = py_match.group(1) if py_match else '3'
        account_id = os.environ.get('ACCOUNT_ID')
        region = os.environ.get('REGION', 'us-east-1')        
        # sagemaker-local test
        base_cmd = [
            "python3", "-m", "pytest", "-v", f"integration/{test_type}",
            "--region", region,
            "--docker-base-name", f"{account_id}.dkr.ecr.{region}.amazonaws.com/beta-autogluon-{container_type}",
            "--tag", tag,
            "--framework-version", self.current_version,
            "--processor", processor,
            "--py-version", python_version
        ]
        # sagemaker test (Remote tests)
        if test_type == "sagemaker":
            instance_type = "ml.p3.2xlarge" if processor == "gpu" else "ml.c5.xlarge"
            base_cmd.extend([
                "--aws-id", account_id,
                "--instance-type", instance_type,
                "--sagemaker-regions", region
            ])
        try:
            result = self.run_subprocess_with_logging(base_cmd, capture_output=True, text=True, timeout=1800)
            success = result.returncode == 0
            output = result.stdout + result.stderr
            test_status = 'PASSED' if success else 'FAILED'
            self.logger.info(f"📊 {test_type.upper()} Test {test_status} ({processor.upper()}): {image_uri.split('/')[-1]}")
            return success, output
        except subprocess.TimeoutExpired:
            return False, f"{test_type.upper()} test execution timed out after 30 minutes"
        except Exception as e:
            return False, f"{test_type.upper()} test execution error: {str(e)}"

    def run_sagemaker_test(self, image_uri: str) -> Tuple[bool, str]:
        """Run tests for a specific inference image (both local and sagemaker tests)"""
        all_outputs = []
        all_success = True
        tag = image_uri.split(':')[-1]
        processor = "CPU" if '-cpu-' in tag else "GPU"
        # run both local and sagemaker tests
        test_types = ["local", "sagemaker"]
        for test_type in test_types:
            self.logger.info(f"🎯 Running {test_type.upper()} tests ({processor}): {tag}")
            success, output = self.run_single_test_suite(image_uri, "inference", test_type)
            all_outputs.append(f"\n--- {test_type.upper()} TEST RESULTS ({processor}) ---\n{output}")
            if not success:
                all_success = False
                self.logger.error(f"❌ {test_type.upper()} tests failed ({processor}): {tag}")
            else:
                self.logger.info(f"✅ {test_type.upper()} tests passed ({processor}): {tag}")
        combined_output = "\n".join(all_outputs)
        return all_success, combined_output

    def filter_errors_only(self, test_output:str) -> str:
        """Filter test output to contain only errors, removing warnings"""
        lines=test_output.split('\n')
        error_lines=[]
        for line in lines:
            if any(warn in line.upper() for warn in ['WARNING', 'WARN', 'DEPRECAT', 'FUTURE']):
                continue
            if any(error in line.upper() for error in ['ERROR', 'FAILED', 'EXCEPTION', 'TRACEBACK']):
                error_lines.append(line)
            elif error_lines and line.strip():
                error_lines.append(line)
            elif not line.strip() and error_lines:
                error_lines.append(line)
        return '\n'.join(error_lines)

    def run_sagemaker_test_agent(self) -> bool:
        """Main execution loop for SageMaker inference tests (error reporting only)"""
        original_dir=os.getcwd()
        try:
            latest_images=self.get_latest_ecr_images()
            overall_success=True
            # Setup test environment for inference
            if not self.setup_test_environment():
                return False
            # Process inference images
            for repo, images in latest_images.items():
                for image_uri in images:
                    tag = image_uri.split(':')[-1]
                    processor = "CPU" if '-cpu-' in tag else "GPU"
                    self.logger.info(f"🎯 Testing {processor} inference (Local + SageMaker): {tag}")
                    # Run test once
                    success, output = self.run_sagemaker_test(image_uri)
                    if success:
                        self.logger.info(f"✅ All tests passed for {tag}")
                    else:
                        self.logger.error(f"❌ Tests failed for {tag}")
                        # Filter and display errors only
                        error_output = self.filter_errors_only(output)
                        if error_output.strip():
                            self.logger.error("🔥 ERROR OUTPUT:")
                            self.logger.error(error_output)
                        else:
                            self.logger.info("ℹ️ No significant errors found after filtering")
                        
                        overall_success = False
                    
                    self.test_results[image_uri] = success
                    
            self.print_test_summary()
            return overall_success
        except Exception as e:
            self.logger.error(f"❌ SageMaker Test Agent failed: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def print_test_summary(self):
        """Print test execution summary for inference images only"""
        print("\n" + "="*70)
        print("🧪 SAGEMAKER TEST SUMMARY")
        print("="*70)
        passed=[uri for uri, passed in self.test_results.items() if passed]
        failed=[uri for uri, passed in self.test_results.items() if not passed]
        
        # Separate by processor type (all are inference)
        inference_cpu_passed = [uri for uri in passed if '-cpu-' in uri]
        inference_gpu_passed = [uri for uri in passed if '-gpu-' in uri]
        inference_cpu_failed = [uri for uri in failed if '-cpu-' in uri]
        inference_gpu_failed = [uri for uri in failed if '-gpu-' in uri]
        
        print(f"📊 Total: {len(self.test_results)} | ✅ Passed: {len(passed)} | ❌ Failed: {len(failed)}")
        print(f"CPU: {len(inference_cpu_passed)}✅/{len(inference_cpu_failed)}❌ | GPU: {len(inference_gpu_passed)}✅/{len(inference_gpu_failed)}❌")
        
        if failed:
            print("\n❌ Failed tests:")
            for uri in failed:
                processor = "CPU" if '-cpu-' in uri else "GPU"
                print(f"   - [Inference {processor}] {uri.split('/')[-1]}")
        print("="*70)

def main():
    import argparse
    parser=argparse.ArgumentParser(description='SageMaker Inference Test Agent (Error Reporting Only)')
    parser.add_argument('--current-version', required=True)
    parser.add_argument('--previous-version', required=True)
    parser.add_argument('--fork-url', required=True)
    args=parser.parse_args()
    agent=SageMakerTestAgent(args.current_version, args.previous_version, args.fork_url)
    success=agent.run_sagemaker_test_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()  