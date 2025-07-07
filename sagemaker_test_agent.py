import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import boto3

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from common import BaseAutomation, ECRImageSelector
from automation_logger import LoggerMixin
@dataclass
class TestError:
    error_type:str
    error_message:str
    test_file:str
    line_number:Optional[str]
    full_traceback:str

class TestFixPlan(BaseModel):
    errors:List[Dict]=Field(description="List of test errors found")
    code_fixes:List[Dict]=Field(description="Code fixes to apply")
    config_fixes:List[Dict]=Field(description="Configuration fixes to apply")
    dependency_fixes:List[Dict]=Field(description="Dependency fixes to apply")
    retry_strategy:str=Field(description="Strategy for retrying tests")
    estimated_success_rate:int=Field(description="Estimated success rate after fixes (0-100)")

class SageMakerTestAgent(BaseAutomation,LoggerMixin):
    """Agentic system for automatically running and fixing SageMaker tests"""
    
    def __init__(self, current_version:str, previous_version:str, fork_url:str):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_bedrock_client()
        self.setup_langchain()
        self.test_results={}
        self.setup_logging(current_version,custom_name="sagemaker")

    def setup_bedrock_client(self):
        """Initialize Bedrock client"""
        self.bedrock_client=boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('REGION', 'us-east-1')
        )
        
    def setup_langchain(self):
        """Initialize LangChain with Claude via Bedrock"""
        model_id=os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        inference_profile_arn=os.getenv('BEDROCK_INFERENCE_PROFILE_ARN')
        models_to_try=[inference_profile_arn] if inference_profile_arn else []
        models_to_try.extend([
            model_id,
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ])
        for model in models_to_try:
            if not model:
                continue
            try:
                self.llm=ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model,
                    provider="anthropic" if inference_profile_arn and model == inference_profile_arn else None,
                    model_kwargs={"max_tokens":4000, "temperature":0.1, "top_p":0.9}
                )
                self.logger.info(f"✅ Initialized Bedrock with {model}")
                break
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize {model}:{e}")
                continue
        else:
            raise Exception("Failed to initialize any Bedrock model")
        self.analysis_prompt=ChatPromptTemplate.from_messages([
            ("system", """You are an expert DevOps engineer specializing in SageMaker testing and AutoGluon.

            Analyze SageMaker test failures and create fix plans to resolve ERRORS (ignore warnings).

            Return ONLY valid JSON with this structure:
            {{
              "errors":[{{"package":"module_name", "description":"brief error description"}}],
              "code_fixes":[{{"type":"modify_file", "file_path":"path/to/file", "changes":"description", "new_content":"full file content"}}],
              "config_fixes":[{{"type":"config_change", "setting":"SETTING_NAME", "value":"new_value", "description":"why"}}],
              "dependency_fixes":[{{"type":"dependency", "action":"install", "package":"package_name", "version":"version"}}],
              "retry_strategy":"modified",
              "estimated_success_rate":85
            }}"""),
            ("human", """Analyze this SageMaker test failure:

            Container:{container_type}, Tag:{image_tag}, Framework:{framework_version}
            Test Directory:{test_directory}

            Errors:
            {test_output}

            Return JSON fix plan.""")
        ])
        
        self.parser=JsonOutputParser(pydantic_object=TestFixPlan)
        self.chain=self.analysis_prompt | self.llm | self.parser

    def get_latest_ecr_images(self) -> Dict[str, List[str]]:
        """Get latest 1 CPU image from beta-autogluon repositories"""
        account_id=os.environ.get('ACCOUNT_ID')
        region=os.environ.get('REGION', 'us-east-1')
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set")
        ecr_client=boto3.client('ecr', region_name=region)
        repositories=['beta-autogluon-training', 'beta-autogluon-inference']
        latest_images={}
        for repo in repositories:
            try:
                response=ecr_client.describe_images(repositoryName=repo, maxResults=50)
                images=sorted(response['imageDetails'], key=lambda x:x['imagePushedAt'], reverse=True)
                for image in images:
                    if 'imageTags' in image:
                        tag=image['imageTags'][0]
                        if '-cpu-' in tag:
                            image_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
                            latest_images[repo]=[image_uri]
                            self.logger.info(f"📦 {repo}:{tag}")
                            break
                else:
                    latest_images[repo]=[]
            except Exception as e:
                self.logger.error(f"❌ Failed to get images from {repo}:{e}")
                latest_images[repo]=[]
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

    def setup_test_environment(self, container_type:str) -> bool:
        """Setup test environment"""
        test_dir=self.repo_dir / f"test/sagemaker_tests/autogluon/{container_type}"
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
            self.logger.info(f"✅ Setup complete for {container_type}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Setup failed:{e}")
            return False

    def run_sagemaker_test(self, image_uri:str, container_type:str) -> Tuple[bool, str]:
        """Run SageMaker test for a specific image"""
        tag=image_uri.split(':')[-1]
        processor='cpu' if '-cpu-' in tag else 'gpu'
        py_match=re.search(r'-py(\d+)-', tag)
        python_version=py_match.group(1) if py_match else '3'
        account_id=os.environ.get('ACCOUNT_ID')
        region=os.environ.get('REGION', 'us-east-1')
        cmd=[
            "python3", "-m", "pytest", "-v", "integration/local",
            "--region", region,
            "--docker-base-name", f"{account_id}.dkr.ecr.{region}.amazonaws.com/beta-autogluon-{container_type}",
            "--tag", tag,
            "--framework-version", self.current_version,
            "--processor", processor,
            "--py-version", python_version
        ]
        try:
            result=self.run_subprocess_with_logging(cmd, capture_output=True, text=True, timeout=1800)
            success=result.returncode == 0
            output=result.stdout + result.stderr
            self.logger.info(f"📊 Test {'PASSED' if success else 'FAILED'}:{image_uri.split('/')[-1]}")
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out after 30 minutes"
        except Exception as e:
            return False, f"Test execution error:{str(e)}"

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

    def apply_fixes(self, fix_plan:Dict, test_directory:str) -> bool:
        """Apply all fixes from Claude's plan"""
        fixes_applied=False
        for fix in fix_plan.get('code_fixes', []):
            if fix.get('type') == 'modify_file':
                file_path=Path(test_directory) / fix.get('file_path', '')
                if not file_path.exists() and 'src/' in fix.get('file_path', ''):
                    file_path=self.repo_dir / fix.get('file_path', '')
                if file_path.exists():
                    file_path.rename(file_path.with_suffix(file_path.suffix + '.backup'))
                    with open(file_path, 'w') as f:
                        f.write(fix.get('new_content', ''))
                    fixes_applied=True
                    self.logger.info(f"✅ Modified:{file_path.name}")
        for fix in fix_plan.get('config_fixes', []):
            if fix.get('type') == 'config_change':
                os.environ[fix.get('setting', '')]=fix.get('value', '')
                fixes_applied=True
                self.logger.info(f"✅ Set {fix.get('setting')}")
        for fix in fix_plan.get('dependency_fixes', []):
            if fix.get('type') == 'dependency':
                package=fix.get('package', '')
                version=fix.get('version', '')
                package_spec=package if version in ['latest', ''] else f"{package}=={version}"
                try:
                    self.run_subprocess_with_logging(["pip3", "install", package_spec], check=True)
                    fixes_applied=True
                    self.logger.info(f"✅ Installed:{package}")
                except subprocess.CalledProcessError:
                    self.logger.warning(f"⚠️ Could not install:{package}")
        return fixes_applied

    def run_sagemaker_test_agent(self) -> bool:
        """Main agent execution loop for SageMaker tests"""
        self.logger.info("🤖 Starting SageMaker Test Agent...")
        original_dir=os.getcwd()
        try:
            latest_images=self.get_latest_ecr_images()
            overall_success=True
            for repo, images in latest_images.items():
                container_type='training' if 'training' in repo else 'inference'
                if not self.setup_test_environment(container_type):
                    overall_success=False
                    continue
                for image_uri in images:
                    self.logger.info(f"🎯 Testing:{image_uri.split('/')[-1]}")
                    max_retries=3
                    test_passed=False
                    for retry in range(max_retries):
                        success, output=self.run_sagemaker_test(image_uri, container_type)
                        if success:
                            test_passed=True
                            break
                        error_output=self.filter_errors_only(output)
                        if not error_output.strip():
                            test_passed=True  
                            break
                        self.logger.info(f"❌ Found errors (attempt {retry + 1}/{max_retries})")
                        try:
                            tag_info={"tag":image_uri.split(':')[-1]}
                            fix_plan=self.chain.invoke({
                                "container_type":container_type,
                                "image_tag":tag_info['tag'],
                                "framework_version":self.current_version,
                                "test_output":error_output,
                                "test_directory":os.getcwd()
                            })
                            if self.apply_fixes(fix_plan, os.getcwd()):
                                self.logger.info(f"✅ Applied Claude's fixes, retrying...")
                                continue
                            else:
                                self.logger.warning("⚠️ No fixes applied")
                                break
                        except Exception as e:
                            self.logger.error(f"❌ Claude analysis failed:{e}")
                            break
                    if not test_passed:
                        overall_success=False
                    self.test_results[image_uri]=test_passed
                os.chdir(original_dir)
            self.print_test_summary()
            return overall_success
        except Exception as e:
            self.logger.error(f"❌ SageMaker Test Agent failed:{e}")
            return False
        finally:
            os.chdir(original_dir)

    def print_test_summary(self):
        """Print test execution summary"""
        print("\n" + "="*70)
        print("🧪 SAGEMAKER TEST AGENT SUMMARY")
        print("="*70)
        passed=[uri for uri, passed in self.test_results.items() if passed]
        failed=[uri for uri, passed in self.test_results.items() if not passed]
        print(f"📊 Total:{len(self.test_results)} | ✅ Passed:{len(passed)} | ❌ Failed:{len(failed)}")
        if failed:
            print("\n❌ Failed tests:")
            for uri in failed:
                print(f"   - {uri.split('/')[-1]}")
        print("="*70)

def main():
    import argparse
    parser=argparse.ArgumentParser(description='SageMaker Test Agent')
    parser.add_argument('--current-version', required=True)
    parser.add_argument('--previous-version', required=True)
    parser.add_argument('--fork-url', required=True)
    args=parser.parse_args()
    agent=SageMakerTestAgent(args.current_version, args.previous_version, args.fork_url)
    success=agent.run_sagemaker_test_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
