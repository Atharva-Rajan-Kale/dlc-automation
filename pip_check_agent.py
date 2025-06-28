import os
import re
import json
import logging
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import boto3
from datetime import datetime

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from common import BaseAutomation, ECRImageSelector

@dataclass
class DependencyConflict:
    package:str
    installed_version:str
    required_constraint:str
    conflicting_package:str
    fix_action:str

class DependencyFixPlan(BaseModel):
    conflicts:List[Dict]=Field(description="List of dependency conflicts found")
    dockerfile_fixes:List[Dict]=Field(description="Fixes to apply to Dockerfiles")
    pyscan_fixes:List[Dict]=Field(description="Fixes to apply to pyscan if Dockerfile fixes fail")
    rebuild_required:bool=Field(description="Whether rebuild is required")
    use_pyscan_only:bool=Field(description="Whether to skip Dockerfile fixes and go straight to pyscan")

class PipCheckAgent(BaseAutomation):
    """Agentic system for automatically fixing pip check issues on a single image"""
    
    def __init__(self, current_version:str, previous_version:str, fork_url:str):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_bedrock_client()
        self.setup_langchain()
        self.ecr_selector=ECRImageSelector()
        # Predefined list of packages to check for version synchronization
        self.packages_to_sync = [
            # Add your packages here
            "fastai",
            "datasets", 
            "gluonts",
            # Add more packages as needed
        ]
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
        if inference_profile_arn:
            self.logger.info(f"🎯 Using Bedrock inference profile:{inference_profile_arn}")
            try:
                self.llm=ChatBedrock(
                    client=self.bedrock_client,
                    model_id=inference_profile_arn,
                    provider="anthropic",  
                    model_kwargs={
                        "max_tokens":4000,
                        "temperature":0.1,
                        "top_p":0.9,
                    }
                )
                self.logger.info(f"✅ Successfully initialized Bedrock with inference profile")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize with inference profile:{e}")
                self.logger.info("🔄 Falling back to regular model ID...")
                
                self.llm=ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model_id,
                    model_kwargs={
                        "max_tokens":4000,
                        "temperature":0.1,
                        "top_p":0.9,
                    }
                )
        else:
            self.logger.info(f"🎯 Using Bedrock model ID:{model_id}")
            try:
                self.llm=ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model_id,
                    model_kwargs={
                        "max_tokens":4000,
                        "temperature":0.1,
                        "top_p":0.9,
                    }
                )
                self.logger.info(f"✅ Successfully initialized Bedrock with model ID")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Bedrock:{e}")
                
                alternative_models=[
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0"
                ]
                for alt_model in alternative_models:
                    try:
                        self.logger.info(f"🧪 Trying alternative model:{alt_model}")
                        self.llm=ChatBedrock(
                            client=self.bedrock_client,
                            model_id=alt_model,
                            model_kwargs={
                                "max_tokens":4000,
                                "temperature":0.1,
                                "top_p":0.9,
                            }
                        )
                        self.logger.info(f"✅ Successfully initialized with {alt_model}")
                        break
                    except Exception as alt_e:
                        self.logger.warning(f"⚠️ {alt_model} also failed:{alt_e}")
                        continue
                else:
                    raise Exception(f"Failed to initialize any Bedrock model. Original error:{e}")
        self.analysis_prompt=ChatPromptTemplate.from_messages([
            ("system", """You are an expert DevOps engineer specializing in Python dependency management and Docker containers.
            Your task is to analyze pip check output and create a comprehensive fix plan for dependency conflicts.           
            STRATEGY:Respect existing package versions, only add NEW dependencies to Dockerfile.           
            IMPORTANT RULES:
            1. If a package is already installed in the Dockerfile (even with conflicting version), DO NOT change it
            2. For existing packages with conflicts, add them to pyscan_fixes to allow the conflict
            3. Only use dockerfile_fixes for completely NEW packages that need to be added
            4. Always set use_pyscan_only:false (try dockerfile for new packages, pyscan for existing conflicts)           
            DOCKERFILE FIXES (only for NEW packages):
            - Format:{{"type":"pin_version", "package":"package_name", "version":"x.y.z"}}
            - Only suggest if the package is NOT already in the current Dockerfile            
            PYSCAN FIXES (for existing package conflicts):
            - Format:{{"package":"package_name", "description":"human-readable explanation"}}
            - Use for any package that already exists in the Dockerfile but has version conflicts
            - Explanation should mention why the existing version is needed/acceptable           
            DECISION PROCESS:
            1. Look at pip conflicts and current Dockerfile content
            2. For each conflicting package:
               - If package EXISTS in Dockerfile → Add to pyscan_fixes
               - If package is MISSING from Dockerfile → Add to dockerfile_fixes  
            3. Set use_pyscan_only:false (we want to try both approaches)           
            Return a JSON object with the exact structure specified in the schema."""),
            ("human", """Analyze this pip check output and create a fix plan:
            Docker Image:{image_tag}
            Container Type:{container_type}
            Pip Check Output:
            {pip_output}           
            Current Dockerfile content (if available):
            {dockerfile_content}           
            Provide a detailed analysis and fix plan. If core dependencies or security-sensitive packages are involved, prefer pyscan over Dockerfile fixes.""")
        ])       
        self.parser=JsonOutputParser(pydantic_object=DependencyFixPlan)
        self.chain=self.analysis_prompt | self.llm | self.parser

    def run_pip_check_on_image(self, image_uri:str) -> Tuple[bool, str]:
        """Run pip check on a Docker image using multiple methods with enhanced timeout handling"""
        self.logger.info(f"🔍 Running pip check on {image_uri}")
        is_inference='inference' in image_uri
        timeout=600 if is_inference else 300  
        if is_inference:
            methods=[
                {
                    "name":"Method 1:Override entrypoint",
                    "cmd":["docker", "run", "--rm", "--entrypoint", "pip", image_uri, "check"]
                },
                {
                    "name":"Method 2:Direct bash approach", 
                    "cmd":["docker", "run", "--rm", image_uri, "bash", "-c", "pip check"]
                },
                {
                    "name":"Method 3:Python module approach",
                    "cmd":["docker", "run", "--rm", image_uri, "python", "-m", "pip", "check"]
                },
                {
                    "name":"Method 4:Simple pip check",
                    "cmd":["docker", "run", "--rm", image_uri, "pip", "check"]
                }
            ]
        else:
            methods=[
                {
                    "name":"Method 1:Direct pip check",
                    "cmd":["docker", "run", "--rm", image_uri, "pip", "check"]
                },
                {
                    "name":"Method 2:Override entrypoint",
                    "cmd":["docker", "run", "--rm", "--entrypoint", "pip", image_uri, "check"]
                },
                {
                    "name":"Method 3:Bash with pip check",
                    "cmd":["docker", "run", "--rm", image_uri, "bash", "-c", "pip check"]
                },
                {
                    "name":"Method 4:Python module pip",
                    "cmd":["docker", "run", "--rm", image_uri, "python", "-m", "pip", "check"]
                }
            ]
        for method in methods:
            try:
                self.logger.info(f"🧪 Trying {method['name']} (timeout:{timeout}s)")
                result=subprocess.run(
                    method['cmd'], 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                output=result.stdout + result.stderr
                self.logger.info(f"📊 Return code:{result.returncode}")
                if output.strip():
                    self.logger.info(f"📝 Output:{output[:200]}...")  
                if 'PIP_CHECK_TIMEOUT' in output:
                    self.logger.warning(f"⏰ Pip check timed out internally")
                    continue
                if result.returncode == 0:
                    if output.strip():
                        
                        self.logger.info(f"✅ Pip check passed for {image_uri} with output")
                        return True, output
                    else:
                        self.logger.info(f"✅ Pip check passed for {image_uri} (no output)")
                        return True, "No dependency conflicts found"
                else:
                    self.logger.warning(f"⚠️ Pip check found conflicts for {image_uri}")
                    return False, output
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⏰ {method['name']} timed out after {timeout}s")
                continue
            except Exception as e:
                self.logger.warning(f"❌ {method['name']} failed:{e}")
                continue
        self.logger.error(f"❌ All pip check methods failed for {image_uri}")
        return False, "All pip check methods failed"

    def parse_dependency_conflicts(self, pip_output:str) -> List[DependencyConflict]:
        """Parse pip check output to extract dependency conflicts"""
        conflicts=[]
        pattern=r'(\S+)\s+([\d.]+)\s+has\s+requirement\s+([^,]+(?:,\s*[^,]+)*),\s+but\s+you\s+have\s+(\S+)\s+([\d.]+)'
        for line in pip_output.split('\n'):
            match=re.search(pattern, line)
            if match:
                conflicting_package=match.group(1)
                conflicting_version=match.group(2)
                requirement=match.group(3)
                package=match.group(4)
                installed_version=match.group(5)
                
                conflict=DependencyConflict(
                    package=package,
                    installed_version=installed_version,
                    required_constraint=requirement,
                    conflicting_package=conflicting_package,
                    fix_action=""
                )
                conflicts.append(conflict)
        return conflicts

    def generate_vulnerability_id(self, package:str, conflict_description:str) -> str:
        """Generate a unique vulnerability ID for pyscan entry"""
        content=f"{package}:{conflict_description}:{self.current_version}"
        hash_obj=hashlib.md5(content.encode())
        return str(int(hash_obj.hexdigest()[:8], 16))[:5]

    def get_pyscan_file_paths(self, container_type:str) -> Dict[str, Path]:
        """Get paths to all pyscan files for a container type"""
        major_minor='.'.join(self.current_version.split('.')[:2])
        base_path=self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
        paths={}
        cpu_path=base_path / "Dockerfile.cpu.py_scan_allowlist.json"
        paths['cpu']=cpu_path
        cuda_dirs=[d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('cu')]
        if cuda_dirs:
            cuda_dir=cuda_dirs[0]  
            gpu_path=cuda_dir / "Dockerfile.gpu.py_scan_allowlist.json"
            paths['gpu']=gpu_path
        else:
            self.logger.warning(f"⚠️ No CUDA directory found in {base_path}")
        return paths

    def load_current_pyscan(self, container_type:str, device_type:str) -> Dict[str, str]:
        """Load current pyscan allowlist for specific container and device type"""
        pyscan_paths=self.get_pyscan_file_paths(container_type)
        pyscan_path=pyscan_paths.get(device_type)
        if not pyscan_path:
            self.logger.warning(f"⚠️ No pyscan path found for {container_type}/{device_type}")
            return {}
        if pyscan_path.exists():
            try:
                with open(pyscan_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load pyscan file {pyscan_path}:{e}")
                return {}
        else:
            self.logger.info(f"ℹ️ Pyscan file not found:{pyscan_path}, will create new one")
            return {}

    def save_pyscan_allowlist(self, container_type:str, device_type:str, allowlist:Dict[str, str]) -> bool:
        """Save updated pyscan allowlist for specific container and device type"""
        try:
            pyscan_paths=self.get_pyscan_file_paths(container_type)
            pyscan_path=pyscan_paths.get(device_type)
            if not pyscan_path:
                self.logger.error(f"❌ No pyscan path found for {container_type}/{device_type}")
                return False
            pyscan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pyscan_path, 'w') as f:
                json.dump(allowlist, f, indent=4, sort_keys=True)
            self.logger.info(f"✅ Updated pyscan allowlist:{pyscan_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save pyscan allowlist for {container_type}/{device_type}:{e}")
            return False

    def get_dockerfile_content(self, container_type:str, device_type:str) -> str:
        """Get current Dockerfile content"""
        major_minor='.'.join(self.current_version.split('.')[:2])
        if device_type == 'cpu':
            dockerfile_path=self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
        else:
            
            py3_dir=self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
            cuda_dirs=[d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
            if cuda_dirs:
                dockerfile_path=cuda_dirs[0] / "Dockerfile.gpu"
            else:
                return ""
        if dockerfile_path.exists():
            return dockerfile_path.read_text()
        return ""

    def apply_dockerfile_fixes(self, container_type:str, fixes:List[Dict]) -> bool:
        """Apply fixes to Dockerfiles - ONLY for NEW packages, never modify existing ones"""
        self.logger.info(f"🔧 Applying Dockerfile fixes for {container_type} (NEW packages only)")
        major_minor='.'.join(self.current_version.split('.')[:2])
        success=True
        for device_type in ['cpu', 'gpu']:
            try:
                if device_type == 'cpu':
                    dockerfile_path=self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
                else:
                    py3_dir=self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
                    cuda_dirs=[d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
                    if not cuda_dirs:
                        continue
                    dockerfile_path=cuda_dirs[0] / "Dockerfile.gpu"
                if not dockerfile_path.exists():
                    continue
                content=dockerfile_path.read_text()
                new_packages=[]
                for fix in fixes:
                    if fix['type'] == 'pin_version':
                        package=fix['package']
                        version=fix['version']
                        if package in content:
                            self.logger.info(f"⚠️ Package {package} already exists in Dockerfile, skipping modification")
                            continue
                        else:
                            new_packages.append((package, version))
                            self.logger.info(f"✅ Package {package} is NEW, will add to Dockerfile")
                if not new_packages:
                    self.logger.info(f"ℹ️ No new packages to add for {device_type}")
                    continue
                lines=content.split('\n')
                fix_lines=[]
                fix_lines.append("#Agent :Adding new dependencies")
                fix_lines.append("RUN pip install --no-cache-dir \\")
                for i, (package, version) in enumerate(new_packages):
                    if i == len(new_packages) - 1: 
                        fix_lines.append(f"    {package}=={version}")
                    else:
                        fix_lines.append(f"    {package}=={version} \\")
                insert_index=len(lines) - 1
                for i, line in enumerate(lines):
                    if line.strip().startswith('COPY') or line.strip().startswith('CMD'):
                        insert_index=i
                        break
                for j, fix_line in enumerate(fix_lines):
                    lines.insert(insert_index + j, fix_line)
                new_content='\n'.join(lines)
                dockerfile_path.write_text(new_content)
                self.logger.info(f"✅ Added {len(new_packages)} NEW packages to {dockerfile_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to update Dockerfile for {device_type}:{e}")
                success=False
        return success

    def trigger_rebuild(self) -> bool:
        """Trigger Step 6 rebuild"""
        self.logger.info("🏗️ Triggering rebuild with Step 6...")
        try:
            from step_6 import Step6Automation
            step6=Step6Automation(self.current_version, self.previous_version, self.fork_url)
            return step6.step6_build_upload_docker()
        except Exception as e:
            self.logger.error(f"❌ Rebuild failed:{e}")
            return False

    def handle_pyscan_fixes(self, container_type:str, fixes:List[Dict]) -> bool:
        """Apply fixes to pyscan allowlists for both CPU and GPU - NO REBUILD AFTER THIS"""
        self.logger.info(f"🔧 Applying pyscan fixes for {container_type} (both CPU and GPU, no rebuild needed)...")
        success=True
        for device_type in ['cpu', 'gpu']:
            try:
                current_allowlist=self.load_current_pyscan(container_type, device_type)
                self.logger.info(f"📋 Current {container_type}/{device_type} pyscan has {len(current_allowlist)} entries")
                for fix in fixes:
                    package=fix.get('package', 'unknown')
                    description=fix.get('description', 'Dependency conflict resolved via pyscan')
                    vuln_id=self.generate_vulnerability_id(package, description)
                    pyscan_entry=f"{package} - {description}"
                    current_allowlist[vuln_id]=pyscan_entry
                    self.logger.info(f"📝 Added {container_type}/{device_type} pyscan entry:{vuln_id}:{pyscan_entry}")
                device_success=self.save_pyscan_allowlist(container_type, device_type, current_allowlist)
                if not device_success:
                    success=False
            except Exception as e:
                self.logger.error(f"❌ Failed to apply pyscan fixes for {container_type}/{device_type}:{e}")
                success=False
        if success:
            self.logger.info(f"✅ Pyscan fixes applied successfully for {container_type} (both CPU and GPU)")
            self.logger.info("ℹ️ Note:Pyscan changes don't require Docker rebuild")
        return success

    def get_latest_ecr_images(self) -> Dict[str, List[str]]:
        """Get latest 2 images from beta-autogluon repositories"""
        self.logger.info("🔍 Getting latest ECR images...")
        account_id=os.environ.get('ACCOUNT_ID')
        region=os.environ.get('REGION', 'us-east-1')
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set")
        ecr_client=boto3.client('ecr', region_name=region)
        repositories=['beta-autogluon-training', 'beta-autogluon-inference']
        latest_images={}
        for repo in repositories:
            try:
                response=ecr_client.describe_images(
                    repositoryName=repo,
                    maxResults=50
                )
                images=sorted(
                    response['imageDetails'], 
                    key=lambda x:x['imagePushedAt'], 
                    reverse=True
                )
                latest_tags=[]
                for image in images[:2]:
                    if 'imageTags' in image:
                        tag=image['imageTags'][0]
                        image_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
                        latest_tags.append(image_uri)
                latest_images[repo]=latest_tags
                self.logger.info(f"📦 {repo}:{latest_tags}")
            except Exception as e:
                self.logger.error(f"❌ Failed to get images from {repo}:{e}")
                latest_images[repo]=[]
        return latest_images
    def update_dockerfile_with_version_range(self, container_type: str, device_type: str, package: str, version_range: str) -> bool:
        """Update Dockerfile to install specific package version range"""
        self.logger.info(f"🔧 Updating {container_type}/{device_type} Dockerfile for {package}=={version_range}")
        
        try:
            major_minor = '.'.join(self.current_version.split('.')[:2])
            
            if device_type == 'cpu':
                dockerfile_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
            else:  # gpu
                py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
                cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
                if not cuda_dirs:
                    self.logger.error(f"❌ No CUDA directory found for {container_type}")
                    return False
                dockerfile_path = cuda_dirs[0] / "Dockerfile.gpu"
            
            if not dockerfile_path.exists():
                self.logger.error(f"❌ Dockerfile not found: {dockerfile_path}")
                return False
            
            content = dockerfile_path.read_text()
            lines = content.split('\n')
            
            # Find the line with autogluon installation
            target_substring = "autogluon==${AUTOGLUON_VERSION}"
            insert_index = -1
            autogluon_line_index = -1
            
            for i, line in enumerate(lines):
                if target_substring in line:
                    insert_index = i + 1
                    autogluon_line_index = i
                    break
            
            if insert_index == -1:
                self.logger.error(f"❌ Could not find autogluon installation line in {dockerfile_path}")
                return False
            
            # Check if autogluon line ends with backslash
            autogluon_line = lines[autogluon_line_index].rstrip()
            if not autogluon_line.endswith('\\'):
                # Add backslash to autogluon line since we're adding something after it
                lines[autogluon_line_index] = autogluon_line + ' \\'
                self.logger.info(f"🔧 Added continuation backslash to autogluon line")
            
            # Determine if our new line should have a backslash
            # Check if there are more RUN continuation lines after the insert point
            needs_backslash = False
            for i in range(insert_index, len(lines)):
                line = lines[i].strip()
                if line and line.startswith('&&'):
                    needs_backslash = True
                    break
                elif line and not line.startswith('&&') and not line.startswith('#'):
                    # Hit a non-continuation, non-comment line
                    break
            
            # Insert the package version update
            if needs_backslash:
                new_line = f" && pip install --no-cache-dir -U \"{package}{version_range}\" \\"
            else:
                new_line = f" && pip install --no-cache-dir -U \"{package}{version_range}\""
            
            lines.insert(insert_index, new_line)
            
            # Write back to file
            new_content = '\n'.join(lines)
            dockerfile_path.write_text(new_content)
            
            self.logger.info(f"✅ Updated {dockerfile_path} with {package}=={version_range}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update Dockerfile for {package}: {e}")
            return False

    def apply_dockerfile_fixes_no_rebuild(self, container_type: str, fixes: List[Dict]) -> bool:
        """Apply fixes to Dockerfiles - ONLY for NEW packages, never modify existing ones (NO REBUILD)"""
        self.logger.info(f"🔧 Applying Dockerfile fixes for {container_type} (NEW packages only, no rebuild)")
        major_minor = '.'.join(self.current_version.split('.')[:2])
        success = True
        
        for device_type in ['cpu', 'gpu']:
            try:
                if device_type == 'cpu':
                    dockerfile_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
                else:
                    py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
                    cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
                    if not cuda_dirs:
                        continue
                    dockerfile_path = cuda_dirs[0] / "Dockerfile.gpu"
                
                if not dockerfile_path.exists():
                    continue
                
                content = dockerfile_path.read_text()
                new_packages = []
                
                for fix in fixes:
                    if fix['type'] == 'pin_version':
                        package = fix['package']
                        version = fix['version']
                        if package in content:
                            self.logger.info(f"⚠️ Package {package} already exists in Dockerfile, skipping modification")
                            continue
                        else:
                            new_packages.append((package, version))
                            self.logger.info(f"✅ Package {package} is NEW, will add to Dockerfile")
                
                if not new_packages:
                    self.logger.info(f"ℹ️ No new packages to add for {device_type}")
                    continue
                
                lines = content.split('\n')
                fix_lines = []
                fix_lines.append("# Agent: Adding new dependencies")
                fix_lines.append("RUN pip install --no-cache-dir \\")
                
                for i, (package, version) in enumerate(new_packages):
                    if i == len(new_packages) - 1:  
                        fix_lines.append(f"    {package}=={version}")
                    else:
                        fix_lines.append(f"    {package}=={version} \\")
                
                insert_index = len(lines) - 1
                for i, line in enumerate(lines):
                    if line.strip().startswith('COPY') or line.strip().startswith('CMD'):
                        insert_index = i
                        break
                
                for j, fix_line in enumerate(fix_lines):
                    lines.insert(insert_index + j, fix_line)
                
                new_content = '\n'.join(lines)
                dockerfile_path.write_text(new_content)
                self.logger.info(f"✅ Added {len(new_packages)} NEW packages to {dockerfile_path} (no rebuild)")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to update Dockerfile for {device_type}: {e}")
                success = False
        
        return success
    
    def get_package_list_from_image(self, image_uri: str) -> Dict[str, str]:
        """Get pip list output from Docker image and return package->version dict"""
        self.logger.info(f"📋 Getting package list from {image_uri}")
        
        methods = [
            {
                "name": "Method 1: Direct pip list",
                "cmd": ["docker", "run", "--rm", image_uri, "pip", "list", "--format=freeze"]
            },
            {
                "name": "Method 2: Override entrypoint",
                "cmd": ["docker", "run", "--rm", "--entrypoint", "pip", image_uri, "list", "--format=freeze"]
            },
            {
                "name": "Method 3: Python module approach",
                "cmd": ["docker", "run", "--rm", image_uri, "python", "-m", "pip", "list", "--format=freeze"]
            }
        ]
        
        for method in methods:
            try:
                self.logger.info(f"🧪 Trying {method['name']} (timeout: 10s)")
                result = subprocess.run(
                    method['cmd'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10  # 10 second timeout for pip list
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    packages = {}
                    for line in result.stdout.strip().split('\n'):
                        if '==' in line:
                            package, version = line.split('==')
                            packages[package.strip()] = version.strip()
                    
                    self.logger.info(f"✅ Found {len(packages)} packages in {image_uri}")
                    return packages
                elif result.stdout.strip():
                    # Even if return code is not 0, if we got output, try to parse it
                    self.logger.info(f"⚠️ Non-zero return code but got output, attempting to parse...")
                    packages = {}
                    for line in result.stdout.strip().split('\n'):
                        if '==' in line:
                            package, version = line.split('==')
                            packages[package.strip()] = version.strip()
                    
                    if packages:
                        self.logger.info(f"✅ Found {len(packages)} packages in {image_uri} (despite non-zero exit)")
                        return packages
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⏰ {method['name']} timed out after 10s (but may have produced output)")
                # For inference images that hang but produce output, we could try to get partial output
                # but subprocess.TimeoutExpired doesn't give us access to partial stdout easily
                continue
            except Exception as e:
                self.logger.warning(f"❌ {method['name']} failed: {e}")
                continue
        
        self.logger.error(f"❌ All pip list methods failed for {image_uri}")
        return {}
    def parse_version_range(self, version: str) -> str:
        """Convert version like '2.8.3' to range like '>=2.8.0,<2.9.0'"""
        try:
            parts = version.split('.')
            if len(parts) >= 2:
                major, minor = parts[0], parts[1]
                next_minor = str(int(minor) + 1)
                return f">={major}.{minor}.0,<{major}.{next_minor}.0"
            else:
                # Fallback for unusual version formats
                return f">={version}"
        except Exception as e:
            self.logger.warning(f"⚠️ Could not parse version {version}: {e}")
            return f">={version}"
    def sync_package_versions(self, latest_images: Dict[str, List[str]]) -> bool:
        """Synchronize package versions between training and inference images"""
        self.logger.info("🔄 Starting package version synchronization...")
        
        try:
            # Get package lists from all images
            image_packages = {}
            
            for repo, images in latest_images.items():
                container_type = 'training' if 'training' in repo else 'inference'
                
                for image_uri in images:
                    # Determine device type from image tag
                    device_type = 'gpu' if 'gpu' in image_uri.lower() or 'cuda' in image_uri.lower() else 'cpu'
                    
                    packages = self.get_package_list_from_image(image_uri)
                    if packages:
                        key = f"{container_type}_{device_type}"
                        image_packages[key] = packages
                        self.logger.info(f"📦 Collected packages for {key}: {len(packages)} packages")
            
            # Compare CPU versions (training vs inference) and update BOTH
            if 'training_cpu' in image_packages and 'inference_cpu' in image_packages:
                cpu_mismatches = self.compare_package_versions(
                    image_packages['training_cpu'], 
                    image_packages['inference_cpu'], 
                    'cpu'
                )
                
                # Update BOTH training AND inference CPU Dockerfiles with the same version range
                for mismatch in cpu_mismatches:
                    target_range = mismatch['target_range']
                    package = mismatch['package']
                    
                    # Update training CPU
                    training_success = self.update_dockerfile_with_version_range(
                        'training', 'cpu', package, target_range
                    )
                    
                    # Update inference CPU with the same range
                    inference_success = self.update_dockerfile_with_version_range(
                        'inference', 'cpu', package, target_range
                    )
                    
                    if training_success and inference_success:
                        self.logger.info(f"✅ Updated both training and inference CPU for {package}={target_range}")
                    elif training_success:
                        self.logger.warning(f"⚠️ Updated training CPU but failed inference CPU for {package}")
                    elif inference_success:
                        self.logger.warning(f"⚠️ Updated inference CPU but failed training CPU for {package}")
                    else:
                        self.logger.error(f"❌ Failed to update both training and inference CPU for {package}")
            
            # Compare GPU versions (training vs inference) and update BOTH
            if 'training_gpu' in image_packages and 'inference_gpu' in image_packages:
                gpu_mismatches = self.compare_package_versions(
                    image_packages['training_gpu'], 
                    image_packages['inference_gpu'], 
                    'gpu'
                )
                
                # Update BOTH training AND inference GPU Dockerfiles with the same version range
                for mismatch in gpu_mismatches:
                    target_range = mismatch['target_range']
                    package = mismatch['package']
                    
                    # Update training GPU
                    training_success = self.update_dockerfile_with_version_range(
                        'training', 'gpu', package, target_range
                    )
                    
                    # Update inference GPU with the same range
                    inference_success = self.update_dockerfile_with_version_range(
                        'inference', 'gpu', package, target_range
                    )
                    
                    if training_success and inference_success:
                        self.logger.info(f"✅ Updated both training and inference GPU for {package}={target_range}")
                    elif training_success:
                        self.logger.warning(f"⚠️ Updated training GPU but failed inference GPU for {package}")
                    elif inference_success:
                        self.logger.warning(f"⚠️ Updated inference GPU but failed training GPU for {package}")
                    else:
                        self.logger.error(f"❌ Failed to update both training and inference GPU for {package}")
            
            self.logger.info("✅ Package version synchronization completed (both training and inference updated)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Package version synchronization failed: {e}")
            return False
    def compare_package_versions(self, training_packages: Dict[str, str], inference_packages: Dict[str, str], device_type: str) -> List[Dict]:
        """Compare package versions between training and inference, return mismatches"""
        mismatches = []
        
        for package in self.packages_to_sync:
            training_version = training_packages.get(package)
            inference_version = inference_packages.get(package)
            
            if training_version and inference_version:
                if training_version != inference_version:
                    self.logger.info(f"🔍 {device_type.upper()} version mismatch for {package}: training={training_version}, inference={inference_version}")
                    
                    # We want to update training to match inference version range
                    version_range = self.parse_version_range(inference_version)
                    mismatches.append({
                        'package': package,
                        'training_version': training_version,
                        'inference_version': inference_version,
                        'target_range': version_range
                    })
            elif training_version and not inference_version:
                self.logger.info(f"ℹ️ {package} exists in training but not in inference ({device_type})")
            elif inference_version and not training_version:
                self.logger.info(f"ℹ️ {package} exists in inference but not in training ({device_type})")
        
        return mismatches
    def run_pip_check_agent(self) -> bool:
        """Main agent execution loop for multiple ECR images"""
        self.logger.info("🤖 Starting Pip Check Agent...")
        
        try:
            # Get latest images from ECR
            latest_images = self.get_latest_ecr_images()
            
            # STEP 1: Synchronize package versions between training and inference (NO REBUILD YET)
            self.logger.info("🔄 Step 1: Synchronizing package versions...")
            sync_success = self.sync_package_versions(latest_images)
            dockerfile_changes_made = sync_success
            
            # STEP 2: Run pip check on current images (existing logic)
            self.logger.info("🔄 Step 2: Running pip check analysis...")
            all_conflicts = {}
            
            for repo, images in latest_images.items():
                container_type = 'training' if 'training' in repo else 'inference'
                
                for image_uri in images:
                    success, output = self.run_pip_check_on_image(image_uri)
                    if not success:
                        conflicts = self.parse_dependency_conflicts(output)
                        if conflicts:
                            all_conflicts[image_uri] = {
                                'container_type': container_type,
                                'conflicts': conflicts,
                                'raw_output': output
                            }
            
            # STEP 3: Handle pip check conflicts (NO REBUILDS DURING THIS PROCESS)
            for image_uri, conflict_data in all_conflicts.items():
                self.logger.info(f"🧠 Analyzing conflicts for {image_uri}")
                container_type = conflict_data['container_type']
                dockerfile_cpu = self.get_dockerfile_content(container_type, 'cpu')
                dockerfile_gpu = self.get_dockerfile_content(container_type, 'gpu')
                dockerfile_content = f"CPU Dockerfile:\n{dockerfile_cpu}\n\nGPU Dockerfile:\n{dockerfile_gpu}"
                
                try:
                    self.logger.info(f"🧠 Sending to Claude: {len(conflict_data['raw_output'])} chars of pip output")
                    fix_plan = self.chain.invoke({
                        "image_tag": image_uri.split('/')[-1],
                        "container_type": container_type,
                        "pip_output": conflict_data['raw_output'],
                        "dockerfile_content": dockerfile_content
                    })
                    
                    self.logger.info(f"🤖 Claude response received: {type(fix_plan)}")
                    self.logger.info(f"📝 Claude response keys: {list(fix_plan.keys()) if isinstance(fix_plan, dict) else 'Not a dict'}")
                    
                    dockerfile_fixes = fix_plan.get('dockerfile_fixes', [])
                    pyscan_fixes = fix_plan.get('pyscan_fixes', [])
                    use_pyscan_only = fix_plan.get('use_pyscan_only', False)
                    
                    self.logger.info(f"📋 Fix plan created: {len(dockerfile_fixes)} Dockerfile fixes, {len(pyscan_fixes)} pyscan fixes")
                    
                    if not dockerfile_fixes and not pyscan_fixes:
                        self.logger.info("🤔 Claude suggested no fixes, creating default pyscan fixes...")
                        pyscan_fixes = []
                        for conflict in conflict_data['conflicts']:
                            pyscan_fixes.append({
                                'package': conflict.package,
                                'description': f"Version conflict: requires {conflict.required_constraint} but have {conflict.installed_version}, resolved via pyscan allowlist"
                            })
                        use_pyscan_only = True
                        self.logger.info(f"📋 Created {len(pyscan_fixes)} default pyscan fixes")
                    
                    # Apply fixes but NO REBUILD YET
                    if use_pyscan_only:
                        self.logger.info(f"🔄 Using pyscan-only strategy for {image_uri}")
                        if pyscan_fixes:
                            if self.handle_pyscan_fixes(container_type, pyscan_fixes):
                                self.logger.info(f"✅ Applied pyscan fixes for {image_uri} (rebuild deferred)")
                            else:
                                self.logger.error(f"❌ Pyscan fixes failed for {image_uri}")
                    else:
                        # Apply Dockerfile fixes first (no rebuild)
                        if dockerfile_fixes:
                            if self.apply_dockerfile_fixes_no_rebuild(container_type, dockerfile_fixes):
                                dockerfile_changes_made = True
                                self.logger.info(f"✅ Applied Dockerfile fixes for {image_uri} (rebuild deferred)")
                            else:
                                self.logger.error(f"❌ Dockerfile fixes failed for {image_uri}")
                        
                        # Apply pyscan fixes as well if needed
                        if pyscan_fixes:
                            if self.handle_pyscan_fixes(container_type, pyscan_fixes):
                                self.logger.info(f"✅ Applied pyscan fixes for {image_uri} (rebuild deferred)")
                            else:
                                self.logger.error(f"❌ Pyscan fixes failed for {image_uri}")
                            
                except Exception as e:
                    self.logger.error(f"❌ Failed to process fix plan for {image_uri}: {e}")
                    self.logger.info(f"🔄 Applying default pyscan fixes as emergency fallback...")
                    emergency_fixes = []
                    for conflict in conflict_data['conflicts']:
                        emergency_fixes.append({
                            'package': conflict.package,
                            'description': f"Emergency fix: {conflict.required_constraint} vs {conflict.installed_version}"
                        })
                    if emergency_fixes:
                        if self.handle_pyscan_fixes(container_type, emergency_fixes):
                            self.logger.info(f"✅ Applied emergency pyscan fixes for {image_uri}")
                        else:
                            self.logger.error(f"❌ Emergency pyscan fixes also failed for {image_uri}")
                    continue
            
            # STEP 4: Final rebuild if any Dockerfile changes were made
            if dockerfile_changes_made:
                self.logger.info("🏗️ Step 4: Triggering final rebuild after all fixes...")
                if self.trigger_rebuild():
                    self.logger.info("✅ Final rebuild completed successfully!")
                else:
                    self.logger.error("❌ Final rebuild failed!")
                    return False
            else:
                self.logger.info("ℹ️ No Dockerfile changes made, skipping rebuild")
            
            if not all_conflicts:
                self.logger.info("✅ No dependency conflicts found!")
            
            self.logger.info("✅ Pip Check Agent completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pip Check Agent failed: {e}")
            return False

def main():
    """Main function for AutoGluon Pip Check Agent"""
    import argparse
    parser=argparse.ArgumentParser(description='AutoGluon Pip Check Agent')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.1)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.0)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    args=parser.parse_args()
    agent=PipCheckAgent(
        args.current_version,
        args.previous_version,
        args.fork_url
    )
    success=agent.run_pip_check_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()