import os
import re
import json
import logging
import subprocess
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import boto3
from datetime import datetime

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from automation.common import BaseAutomation, ECRImageSelector
from automation.automation_logger import LoggerMixin

@dataclass
class DependencyConflict:
    package: str
    installed_version: str
    required_constraint: str
    conflicting_package: str
    conflict_type: str  # 'one_to_one', 'one_to_many', 'platform'

class PipConflictAnalysis(BaseModel):
    conflicts: List[Dict] = Field(description="List of parsed dependency conflicts")
    conflict_groups: Dict[str, List[Dict]] = Field(description="Conflicts grouped by type")

class PipCheckAgent(BaseAutomation,LoggerMixin):
    """Agentic system for automatically fixing pip check issues on a single image"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_bedrock_client()
        self.setup_langchain()
        self.ecr_selector = ECRImageSelector()
        # Predefined list of packages to check for version synchronization
        self.packages_to_sync = [
            "fastai",
            "gluonts",
            # Add more packages as needed
        ]
        # Track failed platform fixes to avoid infinite loops
        self.failed_platform_fixes = set()
        # Track last AI usage time to avoid throttling
        self.last_ai_usage_time = 0
        self.setup_logging(current_version,custom_name="pip_check")

    def wait_for_ai_throttling(self):
        """Wait 3 minutes between AI calls to avoid throttling"""
        current_time = time.time()
        time_since_last_usage = current_time - self.last_ai_usage_time
        wait_time_seconds = 120  # 3 minutes
        if self.last_ai_usage_time > 0 and time_since_last_usage < wait_time_seconds:
            remaining_wait = wait_time_seconds - time_since_last_usage
            self.logger.info(f"⏳ Waiting {remaining_wait:.1f} seconds to avoid AI throttling...")
            # Show countdown in 30-second intervals
            while remaining_wait > 0:
                if remaining_wait > 30:
                    self.logger.info(f"⏳ {remaining_wait:.0f} seconds remaining...")
                    time.sleep(30)
                    remaining_wait -= 30
                else:
                    self.logger.info(f"⏳ {remaining_wait:.0f} seconds remaining...")
                    time.sleep(remaining_wait)
                    remaining_wait = 0
            self.logger.info("✅ Wait complete, proceeding with AI call")
        # Update the last usage time
        self.last_ai_usage_time = time.time()
    
    def setup_bedrock_client(self):
        """Initialize Bedrock client"""
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('REGION', 'us-east-1')
        )
        
    def setup_langchain(self):
        """Initialize LangChain with Claude via Bedrock for conflict parsing only"""
        model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        inference_profile_arn = os.getenv('BEDROCK_INFERENCE_PROFILE_ARN')
        
        if inference_profile_arn:
            self.logger.info(f"🎯 Using Bedrock inference profile: {inference_profile_arn}")
            try:
                self.llm = ChatBedrock(
                    client=self.bedrock_client,
                    model_id=inference_profile_arn,
                    provider="anthropic",  
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                )
                self.logger.info(f"✅ Successfully initialized Bedrock with inference profile")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize with inference profile: {e}")
                self.logger.info("🔄 Falling back to regular model ID...")
                
                self.llm = ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model_id,
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                )
        else:
            self.logger.info(f"🎯 Using Bedrock model ID: {model_id}")
            try:
                self.llm = ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model_id,
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                )
                self.logger.info(f"✅ Successfully initialized Bedrock with model ID")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Bedrock: {e}")
                
                alternative_models = [
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0"
                ]
                for alt_model in alternative_models:
                    try:
                        self.logger.info(f"🧪 Trying alternative model: {alt_model}")
                        self.llm = ChatBedrock(
                            client=self.bedrock_client,
                            model_id=alt_model,
                            model_kwargs={
                                "max_tokens": 4000,
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        )
                        self.logger.info(f"✅ Successfully initialized with {alt_model}")
                        break
                    except Exception as alt_e:
                        self.logger.warning(f"⚠️ {alt_model} also failed: {alt_e}")
                        continue
                else:
                    raise Exception(f"Failed to initialize any Bedrock model. Original error: {e}")

        # Updated prompt for parsing only
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at parsing pip check dependency conflict output.
            Your ONLY task is to parse and categorize the conflicts - do NOT suggest fixes or strategies.
            
            Parse each conflict line and categorize into these types:
            
            1. "one_to_one": Single package has requirement for another single package
               Example: "pathos 0.3.3 has requirement dill>=0.3.9, but you have dill 0.3.8"
               
            2. "platform": Package not supported on platform
               Example: "ninja 1.11.1.1 is not supported on this platform"
               
            3. "one_to_many": Will be determined by grouping logic (multiple conflicts with same package)
            
            For each conflict, extract:
            - conflicting_package: The package that has the requirement
            - package: The package that doesn't meet the requirement  
            - installed_version: Current version installed
            - required_constraint: What version is required
            - conflict_type: one_to_one or platform
            
            Return a JSON object with the exact structure specified in the schema."""),
            ("human", """Parse this pip check output and extract all conflicts:
            
            Pip Check Output:
            {pip_output}
            
            Extract and categorize each conflict. Do not suggest any fixes - only parse the conflicts.""")
        ])       
        
        self.parser = JsonOutputParser(pydantic_object=PipConflictAnalysis)
        self.chain = self.analysis_prompt | self.llm | self.parser

    def run_pip_check_on_image(self, image_uri: str) -> Tuple[bool, str]:
        """Run pip check on a Docker image using multiple methods with enhanced timeout handling"""
        self.logger.info(f"🔍 Running pip check on {image_uri}")
        is_inference = 'inference' in image_uri
        timeout = 600 if is_inference else 300  
        
        if is_inference:
            methods = [
                {
                    "name": "Method 1: Override entrypoint",
                    "cmd": ["docker", "run", "--rm", "--entrypoint", "pip", image_uri, "check"]
                }
            ]
        else:
            methods = [
                {
                    "name": "Method 1: Direct pip check",
                    "cmd": ["docker", "run", "--rm", image_uri, "pip", "check"]
                }
            ]
        
        for method in methods:
            try:
                self.logger.info(f"🧪 Trying {method['name']} (timeout: {timeout}s)")
                result = self.run_subprocess_with_logging(
                    method['cmd'], 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                output = result.stdout + result.stderr
                self.logger.info(f"📊 Return code: {result.returncode}")
                
                if output.strip():
                    self.logger.info(f"📝 Output: {output[:200]}...")  
                
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
                self.logger.warning(f"❌ {method['name']} failed: {e}")
                continue
        
        self.logger.error(f"❌ All pip check methods failed for {image_uri}")
        return False, "All pip check methods failed"

    def parse_conflicts_with_ai(self, pip_output: str) -> List[DependencyConflict]:
        """Use AI to parse pip check output into structured conflicts"""
        try:
            # Wait to avoid throttling
            self.wait_for_ai_throttling()
            
            self.logger.info(f"🧠 Parsing conflicts with AI...")
            parsed_result = self.chain.invoke({
                "pip_output": pip_output
            })
            
            conflicts = []
            for conflict_data in parsed_result.get('conflicts', []):
                conflict = DependencyConflict(
                    package=conflict_data.get('package', ''),
                    installed_version=conflict_data.get('installed_version', ''),
                    required_constraint=conflict_data.get('required_constraint', ''),
                    conflicting_package=conflict_data.get('conflicting_package', ''),
                    conflict_type=conflict_data.get('conflict_type', 'one_to_one')
                )
                conflicts.append(conflict)
            
            self.logger.info(f"✅ Parsed {len(conflicts)} conflicts")
            return conflicts
            
        except Exception as e:
            self.logger.error(f"❌ AI parsing failed: {e}, falling back to regex")
            return self.parse_dependency_conflicts_regex(pip_output)

    def parse_dependency_conflicts_regex(self, pip_output: str) -> List[DependencyConflict]:
        """Fallback regex-based parsing of pip check output"""
        conflicts = []
        # Pattern for dependency conflicts
        dep_pattern = r'(\S+)\s+([\d.]+)\s+has\s+requirement\s+([^,]+(?:,\s*[^,]+)*),\s+but\s+you\s+have\s+(\S+)\s+([\d.]+)'
        # Pattern for platform issues  
        platform_pattern = r'(\S+)\s+([\d.]+)\s+is\s+not\s+supported\s+on\s+this\s+platform'
        for line in pip_output.split('\n'):
            # Check for dependency conflicts
            dep_match = re.search(dep_pattern, line)
            if dep_match:
                conflicting_package = dep_match.group(1)
                conflicting_version = dep_match.group(2)
                requirement = dep_match.group(3)
                package = dep_match.group(4)
                installed_version = dep_match.group(5)
                conflict = DependencyConflict(
                    package=package,
                    installed_version=installed_version,
                    required_constraint=requirement,
                    conflicting_package=conflicting_package,
                    conflict_type="one_to_one"
                )
                conflicts.append(conflict)
                continue
            # Check for platform issues
            platform_match = re.search(platform_pattern, line)
            if platform_match:
                package = platform_match.group(1)
                version = platform_match.group(2)
                conflict = DependencyConflict(
                    package=package,
                    installed_version=version,
                    required_constraint="platform_incompatible",
                    conflicting_package="platform",
                    conflict_type="platform"
                )
                conflicts.append(conflict)
        return conflicts

    def categorize_conflicts(self, conflicts: List[DependencyConflict]) -> Dict[str, List[DependencyConflict]]:
        """Categorize conflicts into one_to_one, one_to_many, and platform"""
        categorized = {
            'one_to_one': [],
            'one_to_many': [],
            'platform': []
        }
        # Group by CONFLICTING package to detect one_to_many (e.g., pathos causing multiple conflicts)
        conflicting_package_groups = {}
        for conflict in conflicts:
            if conflict.conflict_type == "platform":
                categorized['platform'].append(conflict)
            else:
                # Group by the package that's CAUSING the conflicts (e.g., pathos)
                conflicting_pkg = conflict.conflicting_package
                if conflicting_pkg not in conflicting_package_groups:
                    conflicting_package_groups[conflicting_pkg] = []
                conflicting_package_groups[conflicting_pkg].append(conflict)
        # Categorize based on number of conflicts per conflicting package
        for conflicting_package, pkg_conflicts in conflicting_package_groups.items():
            if len(pkg_conflicts) == 1:
                # Single conflict from this package -> one_to_one
                categorized['one_to_one'].extend(pkg_conflicts)
            else:
                # Multiple conflicts from same package -> one_to_many
                for conflict in pkg_conflicts:
                    conflict.conflict_type = "one_to_many"
                categorized['one_to_many'].extend(pkg_conflicts)
                self.logger.info(f"🔍 Detected one-to-many: {conflicting_package} causes {len(pkg_conflicts)} conflicts")
        self.logger.info(f"📊 Conflict categorization: one_to_one={len(categorized['one_to_one'])}, "
                        f"one_to_many={len(categorized['one_to_many'])}, platform={len(categorized['platform'])}")
        return categorized

    def run_compatibility_dry_run(self, image_uri: str, package: str, current_version: str, direction: str, related_packages: List[str]) -> int:
        """Run actual Docker dry run to check how many packages would be affected by version change"""
        self.logger.info(f"🧪 Running compatibility dry run for {package} {direction} {current_version}")
        try:
            if direction == "<":
                constraint = f"{package}<{current_version}"
            else:  # direction == ">"
                constraint = f"{package}>{current_version}"
            self.logger.info(f"🔍 Testing constraint: {constraint}")
            # Use the working command format with -it and bash -c
            cmd = [
                "docker", "run", "--rm", image_uri, "bash", "-c",  # ✅ --rm works in batch
                f"pip install pipdeptree && pip install --dry-run '{constraint}'"
            ]
            self.logger.info(f"🔍 Running command: pip install pipdeptree && pip install --dry-run '{constraint}'")
            result = self.run_subprocess_with_logging(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            output = result.stdout + result.stderr
            self.logger.info(f"📊 Dry run output (first 500 chars): {output[:500]}...")
            # Parse the "Would install" line to count packages
            affected_count = 0
            lines = output.split('\n')
            # First, try to find the "Would install" line
            for line in lines:
                line = line.strip()
                if line.startswith('Would install'):
                    # Handle both formats: "Would install: pkg1 pkg2" and "Would install pkg1 pkg2"
                    if ':' in line:
                        packages_part = line.split(':', 1)[1].strip()
                    else:
                        # Remove "Would install " prefix
                        packages_part = line[len('Would install'):].strip()
                    # Count space-separated package entries
                    package_entries = packages_part.split()
                    affected_count = len(package_entries)
                    self.logger.info(f"📋 Would install packages: {package_entries}")
                    break
            # If we didn't find "Would install" line, try alternative parsing
            if affected_count == 0:
                self.logger.info("📋 'Would install' line not found, trying alternative parsing...")
                # Look for "Downloading" lines as backup (only after pipdeptree install section)
                downloading_packages = set()
                in_dry_run_section = False
                for line in lines:
                    line = line.strip()
                    # Look for the start of the actual dry run (after pipdeptree install)
                    if f"Collecting {package}" in line and package != "pipdeptree":
                        in_dry_run_section = True
                        self.logger.info(f"📋 Found start of dry run section: {line}")
                    if in_dry_run_section and line.startswith('Downloading ') and '.whl' in line:
                        # Extract package name from "Downloading package-1.0.0-py3-none-any.whl"
                        parts = line.split()
                        if len(parts) >= 2:
                            filename = parts[-1]
                            if filename.endswith('.whl') and '-' in filename:
                                package_name = filename.split('-')[0]
                                downloading_packages.add(package_name)
                                self.logger.info(f"📋 Found downloading package: {package_name}")
                # Use the count from downloading packages
                if downloading_packages:
                    affected_count = len(downloading_packages)
                    self.logger.info(f"📋 Total packages to download: {sorted(downloading_packages)}")
                # If still 0, look for the complete output in the last part
                if affected_count == 0:
                    self.logger.info("📋 Still no packages found, checking complete output...")
                    # Print more of the output to debug
                    self.logger.info(f"📊 Complete output: {output}")
                    # Look for any mention of "Would install" even if formatted differently
                    for line in lines:
                        if "would install" in line.lower():
                            self.logger.info(f"📋 Found potential install line: {line}")
                            break
            # If return code indicates error, add penalty
            if result.returncode != 0:
                if affected_count == 0:
                    affected_count = 10  # High penalty for complete failure
                else:
                    affected_count += 2  # Small penalty for warnings/errors but partial success
                self.logger.warning(f"⚠️ Dry run returned error code {result.returncode}")
            self.logger.info(f"📊 Estimated affected packages for {constraint}: {affected_count}")
            return affected_count
        except subprocess.TimeoutExpired:
            self.logger.warning(f"⏰ Compatibility dry run timed out for {constraint}")
            return 999  # High number to indicate failure
        except Exception as e:
            self.logger.error(f"❌ Compatibility dry run failed for {constraint}: {e}")
            return 999  # High number to indicate failure

    def apply_dockerfile_fix(self, container_type: str, package: str, constraint: str) -> bool:
        """Apply package constraint fix to Dockerfiles - avoid duplicates"""
        self.logger.info(f"🔧 Adding {package} {constraint} to {container_type} Dockerfiles")
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
                # Check if package already exists in Dockerfile
                if package in content:
                    self.logger.info(f"⚠️ Package {package} already exists in {dockerfile_path}, skipping to avoid duplicates")
                    continue
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
                    success = False
                    continue
                # Check if autogluon line ends with backslash
                autogluon_line = lines[autogluon_line_index].rstrip()
                if not autogluon_line.endswith('\\'):
                    lines[autogluon_line_index] = autogluon_line + ' \\'
                    self.logger.info(f"🔧 Added continuation backslash to autogluon line")
                # Determine if our new line should have a backslash
                needs_backslash = False
                for i in range(insert_index, len(lines)):
                    line = lines[i].strip()
                    if line and line.startswith('&&'):
                        needs_backslash = True
                        break
                    elif line and not line.startswith('&&') and not line.startswith('#'):
                        break
                # Insert the package constraint
                if needs_backslash:
                    new_line = f" && pip install --no-cache-dir \"{package}{constraint}\" \\"
                else:
                    new_line = f" && pip install --no-cache-dir \"{package}{constraint}\""
                lines.insert(insert_index, new_line)
                # Write back to file
                new_content = '\n'.join(lines)
                dockerfile_path.write_text(new_content)
                self.logger.info(f"✅ Updated {dockerfile_path} with {package}{constraint}")
            except Exception as e:
                self.logger.error(f"❌ Failed to update Dockerfile for {device_type}: {e}")
                success = False
        return success

    def handle_one_to_one_conflicts(self, conflicts: List[DependencyConflict], container_type: str) -> bool:
        """Handle one-to-one dependency conflicts by updating the dependency"""
        self.logger.info(f"🔧 Handling {len(conflicts)} one-to-one conflicts for {container_type}")
        success = True
        for conflict in conflicts:
            try:
                # Extract version constraint
                constraint = conflict.required_constraint
                package = conflict.package
                self.logger.info(f"📦 Updating {package} to meet requirement: {constraint}")
                if self.apply_dockerfile_fix(container_type, package, constraint):
                    self.logger.info(f"✅ Added {package} {constraint} to Dockerfile")
                else:
                    self.logger.error(f"❌ Failed to add {package} {constraint} to Dockerfile")
                    success = False
            except Exception as e:
                self.logger.error(f"❌ Failed to handle one-to-one conflict for {conflict.package}: {e}")
                success = False
        return success

    def handle_one_to_many_conflicts(self, conflicts: List[DependencyConflict], container_type: str, image_uri: str) -> bool:
        """Handle one-to-many conflicts by updating the parent package"""
        self.logger.info(f"🔧 Handling one-to-many conflicts for {container_type}")
        # Group by conflicting package
        package_groups = {}
        for conflict in conflicts:
            pkg = conflict.conflicting_package
            if pkg not in package_groups:
                package_groups[pkg] = []
            package_groups[pkg].append(conflict)
        success = True
        for conflicting_package, pkg_conflicts in package_groups.items():
            try:
                self.logger.info(f"📦 Analyzing {conflicting_package} with {len(pkg_conflicts)} conflicts")
                # Get the version of the CONFLICTING package from the image package list
                packages = self.get_package_list_from_image(image_uri)
                current_version = packages.get(conflicting_package)
                if not current_version:
                    self.logger.error(f"❌ Could not find {conflicting_package} version in image package list")
                    success = False
                    continue
                self.logger.info(f"📋 Found {conflicting_package} version: {current_version}")
                related_packages = [conflict.package for conflict in pkg_conflicts]
                self.logger.info(f"🔍 Related packages: {related_packages}")
                # Run compatibility dry runs for both directions
                less_affected = self.run_compatibility_dry_run(image_uri, conflicting_package, current_version, "<", related_packages)
                greater_affected = self.run_compatibility_dry_run(image_uri, conflicting_package, current_version, ">", related_packages)
                # Choose direction that affects fewer packages
                if less_affected <= greater_affected:
                    constraint = f"<{current_version}"
                    self.logger.info(f"📊 Choosing < direction (affects {less_affected} vs {greater_affected} packages)")
                else:
                    # Parse version and increment for > constraint
                    parts = current_version.split('.')
                    if len(parts) >= 2:
                        minor_version = int(parts[1]) + 1
                        next_version = f"{parts[0]}.{minor_version}.0"
                        constraint = f">={next_version}"
                    else:
                        constraint = f">{current_version}"
                    self.logger.info(f"📊 Choosing > direction (affects {greater_affected} vs {less_affected} packages)")
                
                if self.apply_dockerfile_fix(container_type, conflicting_package, constraint):
                    self.logger.info(f"✅ Added {conflicting_package} {constraint} to Dockerfile")
                else:
                    self.logger.error(f"❌ Failed to add {conflicting_package} {constraint} to Dockerfile")
                    success = False
            except Exception as e:
                self.logger.error(f"❌ Failed to handle one-to-many conflict for {conflicting_package}: {e}")
                success = False
        return success

    def handle_platform_conflicts(self, conflicts: List[DependencyConflict], container_type: str) -> bool:
        """Handle platform conflicts by trying < first, then > if rebuild fails"""
        self.logger.info(f"🔧 Handling {len(conflicts)} platform conflicts for {container_type}")
        success = True
        for conflict in conflicts:
            try:
                package = conflict.package
                version = conflict.installed_version
                # Skip if we already failed with this package
                if package in self.failed_platform_fixes:
                    self.logger.info(f"⚠️ Skipping {package} - already tried both directions")
                    continue
                # Always try < first
                constraint = f"<{version}"
                self.logger.info(f"📦 Trying {package} {constraint} for platform issue")
                if self.apply_dockerfile_fix(container_type, package, constraint):
                    self.logger.info(f"✅ Added {package} {constraint} to Dockerfile (will verify after rebuild)")
                else:
                    self.logger.error(f"❌ Failed to add {package} {constraint} to Dockerfile")
                    success = False
            except Exception as e:
                self.logger.error(f"❌ Failed to handle platform conflict for {conflict.package}: {e}")
                success = False
        return success

    def handle_failed_platform_conflicts(self, conflicts: List[DependencyConflict], container_type: str) -> bool:
        """Handle platform conflicts that failed with < by trying >"""
        self.logger.info(f"🔧 Handling failed platform conflicts with > constraint for {container_type}")
        success = True
        for conflict in conflicts:
            try:
                package = conflict.package
                version = conflict.installed_version
                # Parse version and increment for > constraint
                parts = version.split('.')
                if len(parts) >= 2:
                    minor_version = int(parts[1]) + 1
                    next_version = f"{parts[0]}.{minor_version}.0"
                    constraint = f">={next_version}"
                else:
                    constraint = f">{version}"
                
                self.logger.info(f"📦 Trying {package} {constraint} for platform issue (second attempt)")
                
                if self.apply_dockerfile_fix(container_type, package, constraint):
                    self.logger.info(f"✅ Added {package} {constraint} to Dockerfile")
                else:
                    self.logger.error(f"❌ Failed to add {package} {constraint} to Dockerfile")
                    success = False
                # Mark this package as tried both directions
                self.failed_platform_fixes.add(package)
            except Exception as e:
                self.logger.error(f"❌ Failed to handle failed platform conflict for {conflict.package}: {e}")
                success = False
        return success

    def confirm_rebuild(self) -> bool:
        """Ask user for confirmation before rebuilding"""
        while True:
            try:
                response = input("\n🏗️ Ready to rebuild Docker images. Proceed? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled by user")
                return False

    def trigger_rebuild(self) -> bool:
        """Trigger Step 6 rebuild"""
        self.logger.info("🏗️ Triggering rebuild with Step 6...")
        try:
            from .step_6 import Step6Automation
            step6 = Step6Automation(self.current_version, self.previous_version, self.fork_url)
            return step6.step6_build_upload_docker()
        except Exception as e:
            self.logger.error(f"❌ Rebuild failed: {e}")
            return False

    def generate_vulnerability_id(self, package: str, conflict_description: str) -> str:
        """Generate a unique vulnerability ID for pyscan entry"""
        content = f"{package}:{conflict_description}:{self.current_version}"
        hash_obj = hashlib.md5(content.encode())
        return str(int(hash_obj.hexdigest()[:8], 16))[:5]

    def get_pyscan_file_paths(self, container_type: str) -> Dict[str, Path]:
        """Get paths to all pyscan files for a container type"""
        major_minor = '.'.join(self.current_version.split('.')[:2])
        base_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
        paths = {}
        cpu_path = base_path / "Dockerfile.cpu.py_scan_allowlist.json"
        paths['cpu'] = cpu_path
        cuda_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('cu')]
        if cuda_dirs:
            cuda_dir = cuda_dirs[0]  
            gpu_path = cuda_dir / "Dockerfile.gpu.py_scan_allowlist.json"
            paths['gpu'] = gpu_path
        else:
            self.logger.warning(f"⚠️ No CUDA directory found in {base_path}")
        return paths

    def load_current_pyscan(self, container_type: str, device_type: str) -> Dict[str, str]:
        """Load current pyscan allowlist for specific container and device type"""
        pyscan_paths = self.get_pyscan_file_paths(container_type)
        pyscan_path = pyscan_paths.get(device_type)
        if not pyscan_path:
            self.logger.warning(f"⚠️ No pyscan path found for {container_type}/{device_type}")
            return {}
        if pyscan_path.exists():
            try:
                with open(pyscan_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load pyscan file {pyscan_path}: {e}")
                return {}
        else:
            self.logger.info(f"ℹ️ Pyscan file not found: {pyscan_path}, will create new one")
            return {}

    def save_pyscan_allowlist(self, container_type: str, device_type: str, allowlist: Dict[str, str]) -> bool:
        """Save updated pyscan allowlist for specific container and device type"""
        try:
            pyscan_paths = self.get_pyscan_file_paths(container_type)
            pyscan_path = pyscan_paths.get(device_type)
            if not pyscan_path:
                self.logger.error(f"❌ No pyscan path found for {container_type}/{device_type}")
                return False
            pyscan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pyscan_path, 'w') as f:
                json.dump(allowlist, f, indent=4, sort_keys=True)
            self.logger.info(f"✅ Updated pyscan allowlist: {pyscan_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save pyscan allowlist for {container_type}/{device_type}: {e}")
            return False

    def whitelist_remaining_conflicts(self, conflicts: List[DependencyConflict], container_type: str) -> bool:
        """Whitelist remaining conflicts in pyscan (except platform issues with > constraint)"""
        self.logger.info(f"📝 Whitelisting {len(conflicts)} remaining conflicts for {container_type}")
        success = True
        for device_type in ['cpu', 'gpu']:
            try:
                current_allowlist = self.load_current_pyscan(container_type, device_type)
                self.logger.info(f"📋 Current {container_type}/{device_type} pyscan has {len(current_allowlist)} entries")
                for conflict in conflicts:
                    # Skip platform issues - they should be handled with > constraint
                    if conflict.conflict_type == "platform":
                        self.logger.info(f"⚠️ Skipping platform conflict {conflict.package} from pyscan whitelist")
                        continue
                    package = conflict.package
                    description = f"Conflict: {conflict.conflicting_package} requires {conflict.required_constraint} but have {conflict.installed_version}"
                    vuln_id = self.generate_vulnerability_id(package, description)
                    pyscan_entry = f"{package} - {description}"
                    current_allowlist[vuln_id] = pyscan_entry
                    self.logger.info(f"📝 Added {container_type}/{device_type} pyscan entry: {vuln_id}: {pyscan_entry}")
                device_success = self.save_pyscan_allowlist(container_type, device_type, current_allowlist)
                if not device_success:
                    success = False
            except Exception as e:
                self.logger.error(f"❌ Failed to whitelist conflicts for {container_type}/{device_type}: {e}")
                success = False
        return success

    def get_latest_ecr_images(self) -> Dict[str, List[str]]:
        """Get latest 2 images from beta-autogluon repositories"""
        self.logger.info("🔍 Getting latest ECR images...")
        account_id = os.environ.get('ACCOUNT_ID')
        region = os.environ.get('REGION', 'us-east-1')
        if not account_id:
            raise ValueError("ACCOUNT_ID environment variable not set")
        ecr_client = boto3.client('ecr', region_name=region)
        repositories = ['beta-autogluon-training', 'beta-autogluon-inference']
        latest_images = {}
        for repo in repositories:
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
                latest_tags = []
                for image in images[:2]:
                    if 'imageTags' in image:
                        tag = image['imageTags'][0]
                        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"
                        latest_tags.append(image_uri)
                latest_images[repo] = latest_tags
                self.logger.info(f"📦 {repo}: {latest_tags}")
            except Exception as e:
                self.logger.error(f"❌ Failed to get images from {repo}: {e}")
                latest_images[repo] = []
        return latest_images

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

    def update_dockerfile_with_version_range(self, container_type: str, device_type: str, package: str, version_range: str) -> bool:
        """Update Dockerfile to install specific package version range - avoid duplicates"""
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
            
            # Check if package already exists in Dockerfile
            if package in content:
                self.logger.info(f"⚠️ Package {package} already exists in {dockerfile_path}, skipping to avoid duplicates")
                return True  # Return True since it's not an error, just already exists
            
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
                new_line = f" && pip install --no-cache-dir \"{package}{version_range}\" \\"
            else:
                new_line = f" && pip install --no-cache-dir \"{package}{version_range}\""
            
            lines.insert(insert_index, new_line)
            
            # Write back to file
            new_content = '\n'.join(lines)
            dockerfile_path.write_text(new_content)
            
            self.logger.info(f"✅ Updated {dockerfile_path} with {package}=={version_range}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update Dockerfile for {package}: {e}")
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

    def get_package_list_from_image(self, image_uri: str) -> Dict[str, str]:
        """Get pip list output from Docker image and return package->version dict"""
        self.logger.info(f"📋 Getting package list from {image_uri}")
        timeout = 300  # 5 minute timeout for pip list
        methods = [
            {
                "name": "Method 1: Bash script approach",
                "cmd": ["docker", "run", "--rm", "--entrypoint", "bash", image_uri, 
                       "-c", "pip freeze > /tmp/packages.txt && cat /tmp/packages.txt"]
            },
            {
                "name": "Method 2: Direct pip list",
                "cmd": ["docker", "run", "--rm", image_uri, "pip", "list", "--format=freeze"]
            },
            {
                "name": "Method 3: Override entrypoint",
                "cmd": ["docker", "run", "--rm", "--entrypoint", "pip", image_uri, "list", "--format=freeze"]
            },
            {
                "name": "Method 4: Python module approach",
                "cmd": ["docker", "run", "--rm", image_uri, "python", "-m", "pip", "list", "--format=freeze"]
            }
        ]
        for method in methods:
            try:
                self.logger.info(f"🧪 Trying {method['name']} (timeout: {timeout}s)")
                result = self.run_subprocess_with_logging(
                    method['cmd'], 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
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
                self.logger.warning(f"⏰ {method['name']} timed out after {timeout}s (but may have produced output)")
                # For inference images that hang but produce output, we could try to get partial output
                # but subprocess.TimeoutExpired doesn't give us access to partial stdout easily
                continue
            except Exception as e:
                self.logger.warning(f"❌ {method['name']} failed: {e}")
                continue
        
        self.logger.error(f"❌ All pip list methods failed for {image_uri}")
        return {}

    def run_pip_check_agent(self) -> bool:
        """Main agent execution loop with new manual strategy"""
        self.logger.info("🤖 Starting Pip Check Agent with manual strategy...")
        
        try:
            # Step 1: Get latest images from ECR
            latest_images = self.get_latest_ecr_images()
            
            # Step 2: Synchronize package versions (optional)
            self.logger.info("🔄 Step 1: Synchronizing package versions...")
            sync_success = self.sync_package_versions(latest_images)
            dockerfile_changes_made = sync_success
            
            # Step 3: Run pip check on current images and collect conflicts
            self.logger.info("🔄 Step 2: Running pip check analysis...")
            all_conflicts = {}
            
            for repo, images in latest_images.items():
                container_type = 'training' if 'training' in repo else 'inference'
                
                for image_uri in images:
                    success, output = self.run_pip_check_on_image(image_uri)
                    if not success and "No dependency conflicts found" not in output:
                        # Parse conflicts with AI
                        conflicts = self.parse_conflicts_with_ai(output)
                        if conflicts:
                            all_conflicts[image_uri] = {
                                'container_type': container_type,
                                'conflicts': conflicts,
                                'raw_output': output
                            }
            
            if not all_conflicts:
                self.logger.info("✅ No dependency conflicts found!")
                return True
            
            # Step 4: Process conflicts using manual strategy
            self.logger.info("🔄 Step 3: Processing conflicts with manual strategy...")
            
            for image_uri, conflict_data in all_conflicts.items():
                container_type = conflict_data['container_type']
                conflicts = conflict_data['conflicts']
                
                self.logger.info(f"🔧 Processing {len(conflicts)} conflicts for {image_uri}")
                
                # Categorize conflicts
                categorized = self.categorize_conflicts(conflicts)
                
                # Handle each type of conflict
                if categorized['one_to_one']:
                    if self.handle_one_to_one_conflicts(categorized['one_to_one'], container_type):
                        dockerfile_changes_made = True
                        self.logger.info(f"✅ Handled one-to-one conflicts for {container_type}")
                    else:
                        self.logger.error(f"❌ Failed to handle one-to-one conflicts for {container_type}")
                
                if categorized['one_to_many']:
                    if self.handle_one_to_many_conflicts(categorized['one_to_many'], container_type, image_uri):
                        dockerfile_changes_made = True
                        self.logger.info(f"✅ Handled one-to-many conflicts for {container_type}")
                    else:
                        self.logger.error(f"❌ Failed to handle one-to-many conflicts for {container_type}")
                
                if categorized['platform']:
                    if self.handle_platform_conflicts(categorized['platform'], container_type):
                        dockerfile_changes_made = True
                        self.logger.info(f"✅ Handled platform conflicts for {container_type}")
                    else:
                        self.logger.error(f"❌ Failed to handle platform conflicts for {container_type}")
            
            # Step 5: Rebuild if changes were made
            if dockerfile_changes_made:
                self.logger.info("🏗️ Step 4: Rebuilding Docker images...")
                if not self.trigger_rebuild():
                    self.logger.error("❌ Rebuild failed!")
                    return False
                
                # Step 6: Re-run pip check after rebuild
                self.logger.info("🔄 Step 5: Re-running pip check after rebuild...")
                post_rebuild_conflicts = {}
                
                # Get new images after rebuild (could be the same URIs or new ones)
                latest_images_post = self.get_latest_ecr_images()
                
                for repo, images in latest_images_post.items():
                    container_type = 'training' if 'training' in repo else 'inference'
                    
                    for image_uri in images:
                        success, output = self.run_pip_check_on_image(image_uri)
                        if not success and "No dependency conflicts found" not in output:
                            conflicts = self.parse_conflicts_with_ai(output)
                            if conflicts:
                                post_rebuild_conflicts[image_uri] = {
                                    'container_type': container_type,
                                    'conflicts': conflicts
                                }
                
                # Step 7: Handle remaining conflicts
                if post_rebuild_conflicts:
                    self.logger.info("🔄 Step 6: Handling remaining conflicts...")
                    
                    for image_uri, conflict_data in post_rebuild_conflicts.items():
                        container_type = conflict_data['container_type']
                        conflicts = conflict_data['conflicts']
                        
                        categorized = self.categorize_conflicts(conflicts)
                        
                        # Handle failed platform conflicts with > constraint
                        platform_conflicts = categorized['platform']
                        if platform_conflicts:
                            self.logger.info(f"🔧 Applying > constraint for failed platform conflicts")
                            self.handle_failed_platform_conflicts(platform_conflicts, container_type)
                        
                        # Whitelist remaining conflicts (except platform ones)
                        all_remaining = categorized['one_to_one'] + categorized['one_to_many']
                        if all_remaining:
                            self.whitelist_remaining_conflicts(all_remaining, container_type)
                
                self.logger.info("✅ Pip Check Agent completed successfully!")
                return True
                
            else:
                self.logger.info("ℹ️ No Dockerfile changes made, skipping rebuild")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Pip Check Agent failed: {e}")
            return False

def main():
    """Main function for AutoGluon Pip Check Agent"""
    import argparse
    parser = argparse.ArgumentParser(description='AutoGluon Pip Check Agent')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.1)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.0)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    args = parser.parse_args()
    
    agent = PipCheckAgent(
        args.current_version,
        args.previous_version,
        args.fork_url
    )
    
    success = agent.run_pip_check_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()