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
            
            if training_success and inference_success:
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
        python_version=f"py{python_match.group(1)}" if python_match else "py311"
        os_match=re.search(r'ubuntu(\d+\.\d+)', sample_image.tag)
        os_version=f"ubuntu{os_match.group(1)}" if os_match else "ubuntu22.04"
        cuda_version=self.selected_images['cuda_version']
        pytorch_version=self.selected_images['pytorch_version']
        return {
            'python_version': python_version,
            'os_version': os_version, 
            'cuda_version': cuda_version,
            'pytorch_version': pytorch_version
        }

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
        """Update a single Dockerfile with new FROM statement and AUTOGLUON_VERSION"""
        try:
            with open(dockerfile_path, 'r') as f:
                content=f.read()
            original_content=content
            content=re.sub(
                r'^FROM\s+.*$',
                f'FROM {new_image_uri}',
                content,
                flags=re.MULTILINE
            )
            content=re.sub(
                r'AUTOGLUON_VERSION=[\d.]+',
                f'AUTOGLUON_VERSION={self.current_version}',
                content
            )
            if content != original_content:
                with open(dockerfile_path, 'w') as f:
                    f.write(content)
                self.logger.info(f"✅ Updated {image_type} Dockerfile: {dockerfile_path.relative_to(Path('.'))}")
                self.logger.info(f"   FROM: {new_image_uri}")
            else:
                self.logger.info(f"ℹ️  No changes needed for {dockerfile_path.relative_to(Path('.'))}")
        except Exception as e:
            self.logger.error(f"❌ Failed to update {dockerfile_path}: {e}")

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