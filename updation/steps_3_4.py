import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, List, Any
from automation.common import BaseAutomation, ECRImageSelector
from automation.automation_logger import LoggerMixin

class Steps34Automation(BaseAutomation,LoggerMixin):
    """Handles Steps 3 and 4: Docker resources and Buildspec updates"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.selected_images=None
        self.package_exceptions=None
        self.setup_logging(current_version,custom_name="steps_3_4")
    
    def _load_package_exceptions(self) -> Dict[str, Any]:
        """Load package exceptions from exceptions.txt file"""
        if self.package_exceptions is not None:
            return self.package_exceptions
        # Look for exceptions.txt in the same directory as this script
        script_dir = Path(__file__).parent
        exceptions_file = script_dir / "exceptions.txt"
        
        if not exceptions_file.exists():
            self.logger.info(f"No exceptions.txt file found at {exceptions_file}, no packages will be preserved after autogluon installation")
            self.package_exceptions = {'individual_packages': set(), 'training_additions': [], 'inference_additions': []}
            return self.package_exceptions
            
        try:
            with open(exceptions_file, 'r') as f:
                lines = f.readlines()
                
            individual_packages = set()
            training_additions = []
            inference_additions = []
            current_section = 'individual'
            
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    if '# TRAINING EXCEPTIONS' in line:
                        current_section = 'training'
                        continue
                    elif '# INFERENCE EXCEPTIONS' in line:
                        current_section = 'inference'
                        continue
                    elif stripped_line.startswith('#'):
                        continue
                
                if current_section == 'individual' and stripped_line:
                    package_name = stripped_line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                    individual_packages.add(package_name)
                elif current_section == 'training' and stripped_line:
                    training_additions.append(line.rstrip() + '\n') 
                elif current_section == 'inference' and stripped_line:
                    inference_additions.append(line.rstrip() + '\n')
                        
            self.package_exceptions = {
                'individual_packages': individual_packages,
                'training_additions': training_additions,
                'inference_additions': inference_additions
            }
            return self.package_exceptions
            
        except Exception as e:
            self.logger.error(f"Error reading {exceptions_file}: {e}")
            self.package_exceptions = {'individual_packages': set(), 'training_additions': [], 'inference_additions': []}
            return self.package_exceptions
    
    def _extract_package_names_from_line(self, line: str) -> Set[str]:
        """Extract package names from a pip install line"""
        packages = set()
        line_clean = re.sub(r'^\s*&&\s*pip\s+install\s+.*?(?:--no-cache-dir\s+|--trusted-host\s+\S+\s+)*', '', line)
        line_clean = re.sub(r'^\s*pip\s+install\s+.*?(?:--no-cache-dir\s+|--trusted-host\s+\S+\s+)*', '', line_clean)
        parts = line_clean.split()
        for part in parts:
            part = part.strip().strip('"\'').strip('\\')
            if part and not part.startswith('-') and part != '&&':
                package_name = re.split(r'[<>=!]', part)[0].strip()
                if package_name:
                    packages.add(package_name)
        return packages
    
    def _should_keep_line_after_autogluon(self, line: str) -> Tuple[bool, bool]:
        """Check if a line after autogluon installation should be kept based on exceptions
        Returns: (should_keep, is_last_individual_package)"""
        stripped_line = line.strip()
        
        if 'pip install' not in stripped_line:
            return True, False
            
        package_names = self._extract_package_names_from_line(line)
        exceptions = self._load_package_exceptions()
        individual_packages = exceptions['individual_packages']
        
        should_keep = bool(package_names.intersection(individual_packages))
        is_last_individual = False
        if should_keep:
            found_packages = package_names.intersection(individual_packages)
            last_package = max(individual_packages) if individual_packages else None
            if last_package and last_package in found_packages:
                is_last_individual = True
            
        return should_keep, is_last_individual

    def step3_create_docker_resources(self):
        """Step 3: Create new release docker resources"""
        self.logger.info("Step 3: Creating docker resources")
        original_dir=os.getcwd()
        try:
            if not self.repo_dir.exists():
                self.logger.error(f"Repository directory not found: {self.repo_dir}")
                return False
            os.chdir(self.repo_dir)
            self.logger.info(f"Current working directory: {os.getcwd()}")
            current_files=list(Path(".").iterdir())
            self.logger.info(f"Files in current directory: {[f.name for f in current_files]}")
            if not Path("autogluon").exists():
                self.logger.error("autogluon directory not found in current directory")
                return False
            self.logger.info("Selecting optimal base images from ECR...")
            ecr_selector=ECRImageSelector()
            image_selection=ecr_selector.select_matching_cuda_images()
            if not image_selection:
                self.logger.error("Could not find matching CUDA images")
                return False
            pytorch_version=image_selection['pytorch_version']
            cuda_version=image_selection['cuda_version']
            self.logger.info(f"Selected PyTorch version: {pytorch_version}")
            self.logger.info(f"Selected CUDA version: {cuda_version}")
            self.logger.info(f"Training CPU: {image_selection['training_cpu'].tag}")
            self.logger.info(f"Training GPU: {image_selection['training_gpu'].tag}")
            self.logger.info(f"Inference CPU: {image_selection['inference_cpu'].tag}")
            self.logger.info(f"Inference GPU: {image_selection['inference_gpu'].tag}")
            self.selected_images=image_selection
            if self.is_major_release:
                success=self.create_major_docker_resources(image_selection)
            else:
                success=self.update_minor_docker_resources(image_selection)
            if success:
                self.logger.info("✅ Step 3 completed: Docker resources created (not committed)")
            else:
                self.logger.error("❌ Step 3 failed: Could not create docker resources")
            return success
        except Exception as e:
            self.logger.error(f"❌ Step 3 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def step4update_buildspec_files(self):
        """Step 4: Update buildspec.yml files"""
        self.logger.info("Step 4: Updating buildspec files")
        
        if not self.selected_images:
            self.logger.error("No selected images from step 3. Run step 3 first.")
            return False
        original_dir=os.getcwd()
        try:
            if not self.repo_dir.exists():
                self.logger.error(f"Repository directory not found: {self.repo_dir}")
                return False
            
            os.chdir(self.repo_dir)
            self.logger.info(f"Current working directory: {os.getcwd()}")
            
            image_info=self.extract_buildspec_info_from_images()
            self.logger.info(f"Extracted image info: {image_info}")
            
            training_success=self.update_buildspec("training", image_info)
            inference_success=self.update_buildspec("inference", image_info)
            conftest_success=self.update_conftest_py_version(image_info)
            if training_success and inference_success and conftest_success:
                self.logger.info("✅ Step 4 completed: Buildspec files updated (not committed)")
                return True
            else:
                self.logger.error("❌ Step 4 failed: Could not update buildspec files")
                return False
        except Exception as e:
            self.logger.error(f"❌ Step 4 failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def extract_buildspec_info_from_images(self):
        """Extract version info from selected images for buildspec updates"""
        sample_image=self.selected_images['training_cpu']
        self.logger.info(f"Extracting info from sample image tag: {sample_image.tag}")
        python_match=re.search(r'-py(\d+)-', sample_image.tag)
        python_version=f"py{python_match.group(1)}"
        os_match=re.search(r'ubuntu(\d+\.\d+)', sample_image.tag)
        os_version=f"ubuntu{os_match.group(1)}"
        cuda_version=self.selected_images['cuda_version']
        pytorch_version=self.selected_images['pytorch_version']
        return {
            'python_version': python_version,
            'os_version': os_version, 
            'cuda_version': cuda_version,
            'pytorch_version': pytorch_version
        }
    
    def update_conftest_py_version(self, image_info):
        """Update conftest.py with the extracted python version"""
        conftest_path = Path("test/sagemaker_tests/autogluon/inference/conftest.py")
        if not conftest_path.exists():
            self.logger.error(f"Conftest file not found: {conftest_path}")
            return False
        self.logger.info(f"Updating conftest.py: {conftest_path}")
        # Extract numeric python version (remove 'py' prefix)
        python_version_numeric = image_info['python_version'][2:]
        self.logger.info(f"Updating conftest.py with python version: {python_version_numeric}")
        try:
            with open(conftest_path, 'r') as f:
                content = f.read()
            original_content = content
            # Extract current choices and update them if needed
            choices_pattern = r'choices=\[([^\]]+)\]'
            choices_match = re.search(choices_pattern, content)
            if choices_match:
                # Parse current choices
                choices_str = choices_match.group(1)
                current_choices = [choice.strip().strip('"\'') for choice in choices_str.split(',')]
                self.logger.info(f"Current choices: {current_choices}")
                # Check if current version is different from last choice
                if current_choices and current_choices[-1] != python_version_numeric:
                    # Remove first and append current version
                    new_choices = current_choices[1:] + [python_version_numeric]
                    self.logger.info(f"Updated choices: {new_choices} (removed first, added {python_version_numeric})")
                else:
                    new_choices = current_choices
                    self.logger.info(f"No changes to choices needed (current version {python_version_numeric} matches last choice)")
            else:
                if new_choices[-1] != python_version_numeric:
                    new_choices = new_choices[1:] + [python_version_numeric]
                self.logger.info(f"No existing choices found, using: {new_choices}")
            choices_formatted = ', '.join([f'"{choice}"' for choice in new_choices])
            # Update the parser.addoption line with new choices and default
            pattern = r'parser\.addoption\("--py-version",\s*choices=\[.*?\],\s*default=".*?"\)'
            replacement = f'parser.addoption("--py-version", choices=[{choices_formatted}], default="{python_version_numeric}")'
            # Update the content
            updated_content = re.sub(pattern, replacement, content)
            if updated_content != original_content:
                with open(conftest_path, 'w') as f:
                    f.write(updated_content)
                self.logger.info(f"✅ Successfully updated {conftest_path}")
                self.logger.info(f"   📋 Updated choices to: {new_choices}")
                self.logger.info(f"   📋 Updated python version default to: {python_version_numeric}")
                return True
            else:
                self.logger.info(f"ℹ️  No changes needed for {conftest_path}")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update {conftest_path}: {e}")
            return False
        
    def update_buildspec(self, container_type, image_info):
        """Update buildspec.yml for training or inference"""
        buildspec_path=Path(f"autogluon/{container_type}/buildspec.yml")
        if not buildspec_path.exists():
            self.logger.error(f"Buildspec not found: {buildspec_path}")
            return False
        self.logger.info(f"Updating buildspec: {buildspec_path}")
        with open(buildspec_path, 'r') as f:
            content=f.read()
        original_content=content
        if self.is_major_release:
            backup_name=f"buildspec-{self.previous_version.replace('.', '-')}.yml"
            backup_path=buildspec_path.parent / backup_name
            with open(backup_path, 'w') as f:
                f.write(content)
            self.logger.info(f"📁 Created backup: {backup_path}")
            curr_short='.'.join(self.current_version.split('.')[:2])  
            content=re.sub(
                r'(short_version:\s*&SHORT_VERSION\s+)[\d.]+',
                rf'\g<1>{curr_short}',
                content
            )
            self.logger.info(f"Updated short_version to: {curr_short}")
        content=re.sub(
            r'(version:\s*&VERSION\s+)[\d.]+',
            rf'\g<1>{self.current_version}',
            content
        )
        self.logger.info(f"Updated version to: {self.current_version}")
        content=re.sub(
            r'(tag_python_version:\s*&TAG_PYTHON_VERSION\s+)\w+',
            rf'\g<1>{image_info["python_version"]}',
            content
        )
        self.logger.info(f"Updated python version to: {image_info['python_version']}")
        content=re.sub(
            r'(cuda_version:\s*&CUDA_VERSION\s+)\w+',
            rf'\g<1>{image_info["cuda_version"]}',
            content
        )
        self.logger.info(f"Updated CUDA version to: {image_info['cuda_version']}")
        content=re.sub(
            r'(os_version:\s*&OS_VERSION\s+)[\w.]+',
            rf'\g<1>{image_info["os_version"]}',
            content
        )
        self.logger.info(f"Updated OS version to: {image_info['os_version']}")
        if content != original_content:
            with open(buildspec_path, 'w') as f:
                f.write(content)
            self.logger.info(f"✅ Successfully updated {buildspec_path}")
            self.logger.info(f"   📋 Changes made to {container_type} buildspec:")
            self.logger.info(f"   - Version: {self.current_version}")
            if self.is_major_release:
                curr_short='.'.join(self.current_version.split('.')[:2])
                self.logger.info(f"   - Short version: {curr_short}")
            self.logger.info(f"   - Python: {image_info['python_version']}")
            self.logger.info(f"   - CUDA: {image_info['cuda_version']}")
            self.logger.info(f"   - OS: {image_info['os_version']}")
        else:
            self.logger.info(f"ℹ️  No changes needed for {buildspec_path}")
        return True
        
    def create_major_docker_resources(self, image_selection: Dict) -> bool:
        """Create new directories for major version (e.g., 1.3 -> 1.4)"""
        self.logger.info("Creating resources for MAJOR version update")
        prev_major_minor='.'.join(self.previous_version.split('.')[:2])  
        curr_major_minor='.'.join(self.current_version.split('.')[:2])   
        source_training_dir=Path(f"autogluon/training/docker/{prev_major_minor}")
        source_inference_dir=Path(f"autogluon/inference/docker/{prev_major_minor}")
        target_training_dir=Path(f"autogluon/training/docker/{curr_major_minor}")
        target_inference_dir=Path(f"autogluon/inference/docker/{curr_major_minor}")
        if not source_training_dir.exists():
            self.logger.error(f"Source training directory does not exist: {source_training_dir}")
            return False
        if not source_inference_dir.exists():
            self.logger.error(f"Source inference directory does not exist: {source_inference_dir}")
            return False
        self.logger.info(f"Copying {source_training_dir} -> {target_training_dir}")
        self.logger.info(f"Copying {source_inference_dir} -> {target_inference_dir}")
        if target_training_dir.exists():
            shutil.rmtree(target_training_dir)
        if target_inference_dir.exists():
            shutil.rmtree(target_inference_dir)
        shutil.copytree(source_training_dir, target_training_dir)
        shutil.copytree(source_inference_dir, target_inference_dir)
        self.logger.info(f"✅ Copied directory structures for version {curr_major_minor}")
        cuda_num=image_selection['cuda_version'][2:]  
        self.update_dockerfiles_in_directory(target_training_dir, image_selection, "training", cuda_num)
        self.update_dockerfiles_in_directory(target_inference_dir, image_selection, "inference", cuda_num)
        self.logger.info(f"✅ Updated Dockerfiles for version {curr_major_minor}")
        return True

    def update_minor_docker_resources(self, image_selection: Dict) -> bool:
        """Update existing directories for minor version (e.g., 1.3.0 -> 1.3.1)"""
        self.logger.info("Updating resources for MINOR version update")
        major_minor='.'.join(self.current_version.split('.')[:2])  
        training_dir=Path(f"autogluon/training/docker/{major_minor}")
        inference_dir=Path(f"autogluon/inference/docker/{major_minor}")
        if not training_dir.exists():
            self.logger.error(f"Training directory not found: {training_dir}")
            return False
        if not inference_dir.exists():
            self.logger.error(f"Inference directory not found: {inference_dir}")
            return False
        self.logger.info(f"Updating existing directories: {training_dir} and {inference_dir}")
        cuda_num=image_selection['cuda_version'][2:]  
        self.update_dockerfiles_in_directory(training_dir, image_selection, "training", cuda_num)
        self.update_dockerfiles_in_directory(inference_dir, image_selection, "inference", cuda_num)
        self.logger.info(f"✅ Updated Dockerfiles in existing {major_minor} directories")
        return True

    def update_dockerfiles_in_directory(self, base_dir: Path, image_selection: Dict, 
                                        container_type: str, cuda_num: str):
        """Update Dockerfiles in the specified directory structure"""
        gpu_image=image_selection[f'{container_type}_gpu']
        cpu_image=image_selection[f'{container_type}_cpu']
        self.logger.info(f"Updating Dockerfiles in {base_dir}")
        py3_dir=base_dir / "py3"
        if not py3_dir.exists():
            self.logger.error(f"py3 directory not found: {py3_dir}")
            return
        cpu_dockerfile=py3_dir / "Dockerfile.cpu"
        if cpu_dockerfile.exists():
            self.update_single_dockerfile(cpu_dockerfile, cpu_image.image_uri, f"{container_type.upper()} CPU")
        else:
            self.logger.warning(f"CPU Dockerfile not found: {cpu_dockerfile}")
        cuda_dir=py3_dir / f"cu{cuda_num}"
        if cuda_dir.exists():
            gpu_dockerfile=cuda_dir / "Dockerfile.gpu"
            if gpu_dockerfile.exists():
                self.update_single_dockerfile(gpu_dockerfile, gpu_image.image_uri, f"{container_type.upper()} GPU")
            else:
                self.logger.warning(f"GPU Dockerfile not found: {gpu_dockerfile}")
        else:
            existing_cuda_dirs=[d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
            if existing_cuda_dirs:
                old_cuda_dir=existing_cuda_dirs[0]  
                self.logger.info(f"Renaming {old_cuda_dir.name} to cu{cuda_num}")
                old_cuda_dir.rename(cuda_dir)
                gpu_dockerfile=cuda_dir / "Dockerfile.gpu"
                if gpu_dockerfile.exists():
                    self.update_single_dockerfile(gpu_dockerfile, gpu_image.image_uri, f"{container_type.upper()} GPU")
                else:
                    self.logger.warning(f"GPU Dockerfile not found after rename: {gpu_dockerfile}")
            else:
                self.logger.warning(f"No CUDA directory found in {py3_dir}")

    def update_single_dockerfile(self, dockerfile_path: Path, new_image_uri: str, image_type: str):
        """Update a single Dockerfile with FROM statement and AUTOGLUON_VERSION, 
        handling additional dependencies after autogluon installation based on exceptions"""
        try:
            with open(dockerfile_path, 'r') as f:
                lines = f.readlines()
            original_content = ''.join(lines)
            updated_lines = []
            i = 0
            
            # Determine if this is training or inference based on the path
            container_type = 'training' if 'training' in str(dockerfile_path) else 'inference'
            
            while i < len(lines):
                line = lines[i]
                # Update FROM statement
                if line.strip().startswith('FROM '):
                    updated_lines.append(f'FROM {new_image_uri}\n')
                    i += 1
                    continue
                # Update AUTOGLUON_VERSION
                if 'AUTOGLUON_VERSION=' in line:
                    line = re.sub(r'AUTOGLUON_VERSION=[\d.]+', f'AUTOGLUON_VERSION={self.current_version}', line)
                # Handle RUN blocks containing pip install commands
                if line.strip().startswith('RUN ') and 'pip install' in line:
                    # Process this RUN block specially, injecting additions if needed
                    run_block_result = self._process_pip_run_block(lines, i, container_type)
                    updated_lines.extend(run_block_result['processed_lines'])
                    i = run_block_result['next_index']
                    continue
                updated_lines.append(line)
                i += 1
            new_content = ''.join(updated_lines)
            if new_content != original_content:
                with open(dockerfile_path, 'w') as f:
                    f.write(new_content)
                self.logger.info(f"✅ Updated {image_type} Dockerfile: {dockerfile_path.relative_to(Path('.'))}")
                self.logger.info(f"   FROM: {new_image_uri}")
                self.logger.info(f"   AUTOGLUON_VERSION: {self.current_version}")
            else:
                self.logger.info(f"ℹ️  No changes needed for {dockerfile_path.relative_to(Path('.'))}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update {dockerfile_path}: {e}")


    def _process_pip_run_block(self, lines, start_index, container_type):
        """Process a RUN block containing pip install commands, injecting additions and handling exceptions"""
        processed_lines = []
        i = start_index
        found_autogluon = False
        autogluon_line_index = None
        should_stop_processing = False
        
        # Get the appropriate additions for this container type
        exceptions_data = self._load_package_exceptions()
        if container_type == 'training':
            additions = exceptions_data['training_additions']
        else:
            additions = exceptions_data['inference_additions']
        
        # First pass: collect all lines and identify autogluon installation
        temp_lines = []
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            # Check if this line contains the autogluon installation
            if ('autogluon==${AUTOGLUON_VERSION}' in line or 'autogluon==' in line) and 'pip install' in line:
                found_autogluon = True
                autogluon_line_index = len(temp_lines)
                temp_lines.append({'line': line, 'keep': True, 'is_autogluon': True})
            elif found_autogluon and not should_stop_processing:
                if stripped_line.startswith('#'):
                    # Check if this comment precedes a package we want to keep
                    next_line_idx = i + 1
                    should_keep_comment = False
                    if next_line_idx < len(lines):
                        next_line = lines[next_line_idx]
                        if 'pip install' in next_line:
                            should_keep, is_last = self._should_keep_line_after_autogluon(next_line)
                            should_keep_comment = should_keep
                    temp_lines.append({'line': line, 'keep': should_keep_comment, 'is_comment': True})
                elif 'pip install' in stripped_line:
                    should_keep, is_last_individual = self._should_keep_line_after_autogluon(line)
                    temp_lines.append({'line': line, 'keep': should_keep, 'is_pip': True})
                    if is_last_individual:
                        should_stop_processing = True
                        self.logger.info("Reached last individual package - stopping processing of subsequent pip install lines")
                elif (stripped_line and not stripped_line.startswith('&&') and 
                      not stripped_line.startswith('#') and not stripped_line.endswith('\\') and 
                      not stripped_line == ''):
                    # End of RUN block
                    temp_lines.append({'line': line, 'keep': True, 'is_end': True})
                    i += 1
                    break
                else:
                    temp_lines.append({'line': line, 'keep': True})
            elif found_autogluon and should_stop_processing:
                # We've hit the last individual package, skip all subsequent pip install lines
                if 'pip install' in stripped_line:
                    self.logger.info(f"Skipping pip install line after last individual package: {stripped_line}")
                    temp_lines.append({'line': line, 'keep': False, 'is_pip': True})
                elif stripped_line.startswith('#') and i + 1 < len(lines) and 'pip install' in lines[i + 1]:
                    # Skip comments that precede pip install lines
                    self.logger.info(f"Skipping comment before pip install line: {stripped_line}")
                    temp_lines.append({'line': line, 'keep': False, 'is_comment': True})
                elif (stripped_line and not stripped_line.startswith('&&') and 
                      not stripped_line.startswith('#') and not stripped_line.endswith('\\') and 
                      not stripped_line == ''):
                    # End of RUN block
                    temp_lines.append({'line': line, 'keep': True, 'is_end': True})
                    i += 1
                    break
                else:
                    temp_lines.append({'line': line, 'keep': True})
            else:
                temp_lines.append({'line': line, 'keep': True})
                
            i += 1
            
            # Check if we've reached the end of the RUN block
            if not line.rstrip().endswith('\\'):
                continue_run_block = False
                j = i
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    elif next_line.startswith('#'):
                        j += 1
                        continue
                    elif next_line.startswith('&&'):
                        continue_run_block = True
                        break
                    else:
                        break
                if not continue_run_block:
                    break        
        if found_autogluon and additions and autogluon_line_index is not None:
            
            addition_items = []
            for addition_line in additions:
                addition_items.append({'line': addition_line, 'keep': True, 'is_addition': True})
            temp_lines = (temp_lines[:autogluon_line_index] + 
                         addition_items + 
                         temp_lines[autogluon_line_index:])            
            autogluon_line_index += len(addition_items)
        
        kept_lines = [item for item in temp_lines if item['keep']]
        
        for idx, item in enumerate(kept_lines):
            line = item['line']
            is_last_kept = (idx == len(kept_lines) - 1)
            has_next_kept = (idx < len(kept_lines) - 1)
            
            if item.get('is_autogluon', False) or item.get('is_addition', False) or item.get('is_pip', False):
                line = line.rstrip().rstrip('\\').rstrip() + '\n'
                if has_next_kept and not item.get('is_end', False):
                    line = line.rstrip() + ' \\\n'
            
            elif item.get('is_end', False):
                if processed_lines:
                    last_line = processed_lines[-1]
                    if last_line.rstrip().endswith('\\'):
                        processed_lines[-1] = last_line.rstrip().rstrip('\\').rstrip() + '\n'
            
            processed_lines.append(line)
                    
        return {
            'processed_lines': processed_lines,
            'next_index': i
        }
    
    def run_steps(self, steps_only=None):
        """Run steps 3 and 4"""
        results={}
        
        if not steps_only or 3 in steps_only:
            results[3]=self.step3_create_docker_resources()
            
        if not steps_only or 4 in steps_only:
            results[4]=self.step4update_buildspec_files()
        
        return results

    def get_selected_images(self):
        """Return selected images for use by other modules"""
        return self.selected_images

    def set_selected_images(self, selected_images):
        """Set selected images from another module"""
        self.selected_images=selected_images