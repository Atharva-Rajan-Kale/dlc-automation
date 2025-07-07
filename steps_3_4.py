import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional
from common import BaseAutomation, ECRImageSelector
class Steps34Automation(BaseAutomation):
    """Handles Steps 3 and 4: Docker resources and Buildspec updates"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.selected_images=None
    
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
            self.update_single_dockerfile(cpu_dockerfile, cpu_image.image_uri, "CPU")
        else:
            self.logger.warning(f"CPU Dockerfile not found: {cpu_dockerfile}")
        cuda_dir=py3_dir / f"cu{cuda_num}"
        if cuda_dir.exists():
            gpu_dockerfile=cuda_dir / "Dockerfile.gpu"
            if gpu_dockerfile.exists():
                self.update_single_dockerfile(gpu_dockerfile, gpu_image.image_uri, "GPU")
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
                    self.update_single_dockerfile(gpu_dockerfile, gpu_image.image_uri, "GPU")
                else:
                    self.logger.warning(f"GPU Dockerfile not found after rename: {gpu_dockerfile}")
            else:
                self.logger.warning(f"No CUDA directory found in {py3_dir}")

    def update_single_dockerfile(self, dockerfile_path: Path, new_image_uri: str, image_type: str):
        """Update a single Dockerfile with new FROM statement and AUTOGLUON_VERSION, 
        removing additional dependencies after autogluon installation"""
        try:
            with open(dockerfile_path, 'r') as f:
                lines = f.readlines()
            
            original_content = ''.join(lines)
            updated_lines = []
            i = 0
            
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
                
                # Handle RUN blocks containing autogluon installation
                if line.strip().startswith('RUN ') and self._contains_autogluon_install(lines, i):
                    # Process this RUN block specially
                    run_block_result = self._process_autogluon_run_block(lines, i)
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
                self.logger.info(f"   Removed additional dependencies after autogluon installation")
            else:
                self.logger.info(f"ℹ️  No changes needed for {dockerfile_path.relative_to(Path('.'))}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update {dockerfile_path}: {e}")

    def _contains_autogluon_install(self, lines, start_index):
        """Check if a RUN block contains autogluon installation"""
        i = start_index
        while i < len(lines):
            line = lines[i].strip()
            if 'autogluon==${AUTOGLUON_VERSION}' in line or 'autogluon==' in line:
                return True
            # If we hit a line that doesn't continue the RUN command, stop looking
            if line and not line.startswith('&&') and not line.endswith('\\') and not line.startswith('#') and i > start_index:
                break
            i += 1
        return False

    def _process_autogluon_run_block(self, lines, start_index):
        """Process a RUN block containing autogluon, keeping only content up to autogluon install"""
        processed_lines = []
        i = start_index
        found_autogluon = False
        
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            # Add the line by default
            should_add_line = True
            
            # Check if this line contains the autogluon installation
            if ('autogluon==${AUTOGLUON_VERSION}' in line or 'autogluon==' in line) and 'pip install' in line:
                found_autogluon = True
                # Remove continuation character since we're ending the RUN block here
                if line.rstrip().endswith(' \\'):
                    line = line.rstrip().rstrip('\\').rstrip() + '\n'
                elif line.rstrip().endswith('\\'):
                    line = line.rstrip().rstrip('\\') + '\n'
            
            # After finding autogluon, skip all subsequent pip install lines and comments until RUN block ends
            elif found_autogluon:
                # Skip comment lines
                if stripped_line.startswith('#'):
                    should_add_line = False
                # Skip pip install lines that come after autogluon
                elif 'pip install' in stripped_line and (stripped_line.startswith('&&') or stripped_line.startswith('# ')):
                    should_add_line = False
                # Skip continuation lines that are part of pip install commands we're removing
                elif stripped_line.startswith('&&') and 'pip install' in stripped_line:
                    should_add_line = False
                # If this line doesn't continue the RUN command, we've reached the end of the RUN block
                elif (stripped_line and not stripped_line.startswith('&&') and 
                    not stripped_line.startswith('#') and not stripped_line.endswith('\\') and 
                    not stripped_line == ''):
                    # This line starts a new command, add it and break
                    processed_lines.append(line)
                    i += 1
                    break
            
            if should_add_line:
                processed_lines.append(line)
            
            i += 1
            
            # If we've processed the autogluon line and cleaned up the RUN block, check if we should continue
            if found_autogluon and not line.rstrip().endswith('\\'):
                # Look ahead to see if there are more lines in this RUN block
                continue_run_block = False
                j = i
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:  # Empty line
                        j += 1
                        continue
                    elif next_line.startswith('#'):  # Comment line
                        j += 1
                        continue
                    elif next_line.startswith('&&'):  # Continuation of RUN block
                        continue_run_block = True
                        break
                    else:  # New command
                        break
                
                if not continue_run_block:
                    break
        
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