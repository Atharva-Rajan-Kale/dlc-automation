#!/usr/bin/env python3
"""
AutoGluon Release Automation - Setup Validation Script
This script validates that all necessary configurations are in place for GitHub Actions.
"""

import os
import sys
import boto3
import json
import subprocess
import requests
from typing import Dict, List, Tuple, Optional

class SetupValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0

    def log_success(self, message: str):
        print(f"✅ {message}")
        self.success_count += 1

    def log_warning(self, message: str):
        print(f"⚠️  {message}")
        self.warnings.append(message)

    def log_error(self, message: str):
        print(f"❌ {message}")
        self.errors.append(message)

    def check_environment_variables(self) -> bool:
        """Check that all required environment variables are set"""
        print("\n🔍 Checking Environment Variables...")
        self.total_checks += 1
        
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 
            'AWS_ACCOUNT_ID',
            'GITHUB_TOKEN'
        ]
        
        optional_vars = [
            'AWS_REGION',
            'BEDROCK_MODEL_ID',
            'BEDROCK_INFERENCE_PROFILE_ARN',
            'BEDROCK_AWS_ACCESS_KEY_ID',
            'BEDROCK_AWS_SECRET_ACCESS_KEY',
            'BEDROCK_REGION',
            'CODEBUILD_AWS_ACCESS_KEY_ID',
            'CODEBUILD_AWS_SECRET_ACCESS_KEY',
            'CODEBUILD_REGION'
        ]
        
        all_present = True
        
        for var in required_vars:
            if os.environ.get(var):
                self.log_success(f"Required variable {var} is set")
            else:
                self.log_error(f"Required variable {var} is missing")
                all_present = False
        
        for var in optional_vars:
            if os.environ.get(var):
                self.log_success(f"Optional variable {var} is set")
            else:
                self.log_warning(f"Optional variable {var} is not set (will use defaults)")
        
        return all_present

    def check_aws_credentials(self) -> bool:
        """Verify AWS credentials work"""
        print("\n🔍 Checking AWS Credentials...")
        self.total_checks += 1
        
        try:
            # Try to create a session and get caller identity
            session = boto3.Session()
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            
            account_id = identity.get('Account')
            user_arn = identity.get('Arn')
            
            self.log_success(f"AWS credentials valid for account: {account_id}")
            self.log_success(f"Identity: {user_arn}")
            
            # Check if account ID matches environment variable
            env_account_id = os.environ.get('AWS_ACCOUNT_ID')
            if env_account_id and env_account_id == account_id:
                self.log_success("Account ID matches environment variable")
            elif env_account_id:
                self.log_warning(f"Account ID mismatch: env={env_account_id}, actual={account_id}")
            
            return True
            
        except Exception as e:
            self.log_error(f"AWS credentials invalid: {e}")
            return False

    def check_ecr_access(self) -> bool:
        """Check ECR repository access"""
        print("\n🔍 Checking ECR Access...")
        self.total_checks += 1
        
        try:
            region = os.environ.get('AWS_REGION', 'us-east-1')
            ecr = boto3.client('ecr', region_name=region)
            
            # Check access to required repositories
            required_repos = ['beta-autogluon-training', 'beta-autogluon-inference']
            
            for repo in required_repos:
                try:
                    response = ecr.describe_repositories(repositoryNames=[repo])
                    if response['repositories']:
                        self.log_success(f"ECR repository {repo} is accessible")
                    else:
                        self.log_error(f"ECR repository {repo} not found")
                        return False
                except ecr.exceptions.RepositoryNotFoundException:
                    self.log_error(f"ECR repository {repo} does not exist")
                    return False
                except Exception as e:
                    self.log_error(f"Error accessing ECR repository {repo}: {e}")
                    return False
            
            # Test ECR login
            try:
                auth_response = ecr.get_authorization_token()
                if auth_response['authorizationData']:
                    self.log_success("ECR authorization token obtained successfully")
                else:
                    self.log_error("Could not get ECR authorization token")
                    return False
            except Exception as e:
                self.log_error(f"ECR authorization failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"ECR access check failed: {e}")
            return False

    def check_bedrock_access(self) -> bool:
        """Check Bedrock/AI access for security analysis"""
        print("\n🔍 Checking Bedrock Access...")
        self.total_checks += 1
        
        try:
            # Use Bedrock-specific credentials if available
            bedrock_access_key = os.environ.get('BEDROCK_AWS_ACCESS_KEY_ID')
            bedrock_secret_key = os.environ.get('BEDROCK_AWS_SECRET_ACCESS_KEY')
            bedrock_region = os.environ.get('BEDROCK_REGION', 'us-east-1')
            
            if bedrock_access_key and bedrock_secret_key:
                session = boto3.Session(
                    aws_access_key_id=bedrock_access_key,
                    aws_secret_access_key=bedrock_secret_key,
                    region_name=bedrock_region
                )
                self.log_success("Using dedicated Bedrock credentials")
            else:
                session = boto3.Session()
                self.log_warning("Using default AWS credentials for Bedrock")
            
            bedrock = session.client('bedrock-runtime', region_name=bedrock_region)
            
            # Try to list foundation models to verify access
            try:
                # Note: We can't actually invoke the model without costs, 
                # but we can check if the client can be created
                self.log_success("Bedrock client created successfully")
                
                # Check if inference profile ARN is provided
                inference_profile_arn = os.environ.get('BEDROCK_INFERENCE_PROFILE_ARN')
                if inference_profile_arn:
                    self.log_success(f"Inference profile ARN configured: {inference_profile_arn[:50]}...")
                else:
                    self.log_warning("No inference profile ARN configured (will use model ID)")
                
                model_id = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
                self.log_success(f"Model ID configured: {model_id}")
                
                return True
                
            except Exception as e:
                self.log_error(f"Bedrock access verification failed: {e}")
                return False
                
        except Exception as e:
            self.log_error(f"Bedrock setup failed: {e}")
            return False

    def check_codebuild_access(self) -> bool:
        """Check CodeBuild access for log retrieval"""
        print("\n🔍 Checking CodeBuild Access...")
        self.total_checks += 1
        
        try:
            # Use CodeBuild-specific credentials if available
            codebuild_access_key = os.environ.get('CODEBUILD_AWS_ACCESS_KEY_ID')
            codebuild_secret_key = os.environ.get('CODEBUILD_AWS_SECRET_ACCESS_KEY')
            codebuild_region = os.environ.get('CODEBUILD_REGION', 'us-west-2')
            
            if codebuild_access_key and codebuild_secret_key:
                session = boto3.Session(
                    aws_access_key_id=codebuild_access_key,
                    aws_secret_access_key=codebuild_secret_key,
                    region_name=codebuild_region
                )
                self.log_success("Using dedicated CodeBuild credentials")
            else:
                session = boto3.Session()
                self.log_warning("Using default AWS credentials for CodeBuild")
            
            codebuild = session.client('codebuild', region_name=codebuild_region)
            logs = session.client('logs', region_name=codebuild_region)
            
            # Try to list projects (this should work with read permissions)
            try:
                response = codebuild.list_projects(maxResults=1)
                self.log_success("CodeBuild client access verified")
            except Exception as e:
                self.log_warning(f"CodeBuild list projects failed (may not have projects): {e}")
            
            # Try to list log groups
            try:
                response = logs.describe_log_groups(limit=1)
                self.log_success("CloudWatch Logs access verified")
            except Exception as e:
                self.log_warning(f"CloudWatch Logs access failed: {e}")
            
            return True
            
        except Exception as e:
            self.log_error(f"CodeBuild setup failed: {e}")
            return False

    def check_github_token(self) -> bool:
        """Check GitHub token validity"""
        print("\n🔍 Checking GitHub Token...")
        self.total_checks += 1
        
        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            self.log_error("GITHUB_TOKEN not found")
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28'
            }
            
            response = requests.get('https://api.github.com/user', headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                self.log_success(f"GitHub token valid for user: {user_data.get('login', 'unknown')}")
                return True
            else:
                self.log_error(f"GitHub token invalid: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_error(f"GitHub token check failed: {e}")
            return False

    def check_python_dependencies(self) -> bool:
        """Check that required Python packages are available"""
        print("\n🔍 Checking Python Dependencies...")
        self.total_checks += 1
        
        required_packages = [
            'boto3',
            'requests',
            'pyyaml',
            'pathlib'
        ]
        
        ai_packages = [
            'langchain_aws',
            'langchain_core', 
            'pydantic'
        ]
        
        all_present = True
        
        for package in required_packages:
            try:
                __import__(package)
                self.log_success(f"Required package {package} is available")
            except ImportError:
                self.log_error(f"Required package {package} is missing")
                all_present = False
        
        for package in ai_packages:
            try:
                __import__(package)
                self.log_success(f"AI package {package} is available")
            except ImportError:
                self.log_warning(f"AI package {package} is missing (needed for security analysis)")
        
        return all_present

    def check_docker_access(self) -> bool:
        """Check Docker availability"""
        print("\n🔍 Checking Docker Access...")
        self.total_checks += 1
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            self.log_success(f"Docker available: {version}")
            
            # Try to run a simple container
            result = subprocess.run(['docker', 'run', '--rm', 'hello-world'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.log_success("Docker container execution test passed")
            else:
                self.log_warning("Docker container execution test failed")
            
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.log_error(f"Docker not available: {e}")
            return False

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        print("🚀 AutoGluon Release Automation - Setup Validation")
        print("=" * 60)
        
        checks = [
            self.check_environment_variables,
            self.check_python_dependencies,
            self.check_aws_credentials,
            self.check_ecr_access,
            self.check_bedrock_access,
            self.check_codebuild_access,
            self.check_github_token,
            self.check_docker_access
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                self.log_error(f"Check {check.__name__} failed with exception: {e}")
                results.append(False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed_checks = sum(1 for r in results if r)
        total_checks = len(results)
        
        print(f"✅ Passed: {passed_checks}/{total_checks} checks")
        
        if self.warnings:
            print(f"⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.errors:
            print(f"❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   - {error}")
        
        if passed_checks == total_checks and not self.errors:
            print("\n🎉 All checks passed! Your setup is ready for AutoGluon automation.")
            return True
        elif not self.errors:
            print("\n⚠️  Setup is mostly ready, but there are warnings to address.")
            return True
        else:
            print("\n❌ Setup validation failed. Please address the errors above.")
            return False

def main():
    """Main function"""
    validator = SetupValidator()
    success = validator.run_all_checks()
    
    # Set exit code based on validation result
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()