import argparse
import logging
from updation.steps_1_2_5 import Steps125Automation
from updation.steps_3_4 import Steps34Automation  
from updation.step_6 import Step6Automation
from release.step_8_infrastructure import Step8InfrastructureAutomation
from updation.pip_check_agent import PipCheckAgent
from testing.sagemaker_test_agent import SageMakerTestAgent
from testing.security_test_agent import SecurityTestAgent
from testing.github_pr_automation import GitHubPRAutomation
from testing.quick_checks_agent import QuickChecksAgent
from testing.autogluon_test_agent import AutoGluonTestAgent
from release.autogluon_release_automation import AutoGluonReleaseImagesAutomation
from release.asimov_scan_cr import AsimovSecurityScanAutomation

class EnhancedAutoGluonReleaseAutomation:
    """Enhanced orchestrator with complete workflow: Steps 1-7, Infrastructure, Release Images, and Asimov Security Scan"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        self.current_version = current_version
        self.previous_version = previous_version
        self.fork_url = fork_url
        
        current_parts = current_version.split('.')
        previous_parts = previous_version.split('.')
        self.is_major_release = (
            current_parts[0] != previous_parts[0] or 
            (current_parts[1] != previous_parts[1] and current_parts[2] == '0')
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced AutoGluon Release Automation - Version {current_version}")
        self.logger.info(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
        
        # Initialize automation components
        self.steps_125 = Steps125Automation(current_version, previous_version, fork_url)
        self.steps_34 = Steps34Automation(current_version, previous_version, fork_url)
        self.step_6 = Step6Automation(current_version, previous_version, fork_url)
        self.step_8 = Step8InfrastructureAutomation(current_version, previous_version, fork_url)
        self.pip_check_agent = PipCheckAgent(current_version, previous_version, fork_url)
        self.autogluon_test_agent = AutoGluonTestAgent(current_version, previous_version, fork_url)
        self.sagemaker_test_agent = SageMakerTestAgent(current_version, previous_version, fork_url)
        self.security_test_agent = SecurityTestAgent(current_version, previous_version, fork_url)
        self.quick_checks_agent = QuickChecksAgent(current_version, previous_version, fork_url)
        
        # Initialize new automation components
        self.autogluon_release_automation = AutoGluonReleaseImagesAutomation(current_version, previous_version, fork_url)
        self.asimov_scan_automation = AsimovSecurityScanAutomation(current_version, previous_version, fork_url)

        # Initialize PR automation
        self.pr_automation = GitHubPRAutomation(
            current_version=current_version,
            fork_url=fork_url,
            repo_dir=self.steps_125.repo_dir
        )
        
        self.results = {}
        self.selected_images = None

    def step_7_create_pr_with_security(self) -> bool:
        """Step 7: Create Pull Request and run agentic security analysis"""
        self.logger.info("Step 7: Creating PR with agentic security testing")
        try:
            # First create the PR
            pr_success = self.pr_automation.create_pull_request()
            self.results[7] = pr_success
            
            if pr_success:
                self.logger.info("✅ Pull Request created successfully!")
                
                # Get the PR number for waiting
                pr_number = self.security_test_agent.get_current_pr_number()
                
                if pr_number:
                    # Wait for security tests to complete before AI analysis
                    self.logger.info("⏳ Waiting for security tests to complete before AI analysis...")
                    self.security_test_agent.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=45)
                    
                    # Now run agentic security analysis and fixing
                    self.logger.info("🤖 Running Agentic Security Analysis...")
                    security_success = self.security_test_agent.run_deterministic_security_analysis()
                    self.results['security_tests'] = security_success
                    
                    if security_success:
                        self.logger.info("✅ Agentic security analysis completed - all tests passing!")
                    else:
                        self.logger.warning("⚠️ Agentic security analysis found issues that need review")
                    
                    self.logger.info("Running quick checks agent...")
                    quick_checks_success = self.quick_checks_agent.run_quick_checks_agent()
                    self.results['quick_checks'] = quick_checks_success

                    if quick_checks_success:
                        self.logger.info("Quick checks agent completed successfully")
                    else:
                        self.logger.warning("Quick checks agent found issues or was not needed")
                    return pr_success and security_success and quick_checks_success
                else:
                    self.logger.warning("⚠️ Could not get PR number for security test monitoring")
                    return pr_success
            else:
                self.logger.error("❌ Pull Request creation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step 7 with agentic security failed: {e}")
            return False
    
    def step_7_create_pr_with_complete_workflow(self) -> bool:
        """Step 7: Complete workflow - PR + Security + Quick Checks"""
        self.logger.info("Step 7: Creating PR with complete automation workflow")
        
        # First run the normal PR creation with security and quick checks
        pr_success = self.step_7_create_pr_with_security()
        
        if pr_success:
            self.logger.info("✅ PR, security analysis, and quick checks completed")
        else:
            self.logger.error("❌ PR creation failed, skipping all subsequent steps")
            return False

    def run_autogluon_tests_only(self):
        """Run only the AutoGluon tests without any other steps"""
        self.logger.info("🧪 Running AutoGluon Test Agent only...")
        try:
            result = self.autogluon_test_agent.run_autogluon_test_agent()
            self.results['autogluon_tests'] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ AutoGluon Test Agent failed: {e}")
            return False

    def run_release_images_only(self):
        """Run only the AutoGluon Release Images automation"""
        self.logger.info("📦 Running AutoGluon Release Images Automation only...")
        try:
            result = self.autogluon_release_automation.run_release_images_automation()
            self.results['release_images'] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ AutoGluon Release Images automation failed: {e}")
            return False

    def run_asimov_scan_only(self):
        """Run only the Asimov Security Scan automation"""
        self.logger.info("🔒 Running Asimov Security Scan Automation only...")
        try:
            result = self.asimov_scan_automation.run_automation()
            self.results['asimov_scan'] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ Asimov Security Scan automation failed: {e}")
            return False

    def run_post_pr_workflow_only(self):
        """Run only the post-PR workflow: Infrastructure + Release Images + Asimov Scan"""
        self.logger.info("🚀 Running Post-PR Workflow: Infrastructure → Release Images → Asimov Scan...")
        
        try:
            # Set up python version info for infrastructure step
            self.logger.info("🔍 Setting up python version info for infrastructure...")
            if self.selected_images:
                python_info = self.extract_python_version_info()
                self.logger.info(f"🔍 Extracted python_info = {python_info}")
                self.step_8.set_python_version_info(python_info)
                self.logger.info("✅ Python version info passed to Step 8")
            else:
                self.logger.warning("⚠️ No selected images available, Step 8 will use default python version")
                # Still try to set something
                default_info = {'python_version': 'py311'}
                self.step_8.set_python_version_info(default_info)
            
            # Step 1: Run infrastructure deployment (Step 8)
            self.logger.info("🏗️ Running Infrastructure Deployment (Step 8)...")
            infrastructure_success = self.step_8.run_infrastructure_deployment()
            self.results[8] = infrastructure_success
            
            if infrastructure_success:
                self.logger.info("✅ Infrastructure deployment completed successfully!")
                
                # Step 2: Run AutoGluon Release Images automation
                self.logger.info("📦 Running AutoGluon Release Images Automation...")
                release_images_success = self.autogluon_release_automation.run_release_images_automation()
                self.results['release_images'] = release_images_success
                
                if release_images_success:
                    self.logger.info("✅ AutoGluon Release Images automation completed successfully!")
                    
                    # Step 3: Run Asimov Security Scan automation
                    self.logger.info("🔒 Running Asimov Security Scan Automation...")
                    asimov_success = self.asimov_scan_automation.run_automation()
                    self.results['asimov_scan'] = asimov_success
                    
                    if asimov_success:
                        self.logger.info("✅ Asimov Security Scan automation completed successfully!")
                        self.logger.info("🎉 Complete Post-PR Workflow finished successfully!")
                        return True
                    else:
                        self.logger.warning("⚠️ Asimov Security Scan automation failed or had issues")
                        return False
                else:
                    self.logger.warning("⚠️ AutoGluon Release Images automation failed, skipping Asimov scan")
                    return False
            else:
                self.logger.warning("⚠️ Infrastructure deployment failed, skipping subsequent steps")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Post-PR workflow failed: {e}")
            return False
        
    def step_7_create_pr_with_infrastructure(self) -> bool:
        """Step 7: Create PR with security analysis, then proceed to infrastructure deployment (Legacy method)"""
        self.logger.info("Step 7: Creating PR with agentic security testing and infrastructure deployment")
        
        # First run the normal PR creation with security
        pr_success = self.step_7_create_pr_with_security()
        
        if pr_success:
            self.logger.info("✅ PR and security analysis completed, proceeding to infrastructure deployment...")
            
            # Set up python version info for infrastructure step
            self.logger.info(f"🔍 DEBUG: About to extract python info, self.selected_images = {self.selected_images}")
            
            if self.selected_images:
                python_info = self.extract_python_version_info()
                self.logger.info(f"🔍 DEBUG: Extracted python_info = {python_info}")
                self.step_8.set_python_version_info(python_info)
                self.logger.info("✅ Python version info passed to Step 8")
            else:
                self.logger.warning("⚠️ No selected images available, Step 8 will use default python version")
                # Still try to set something
                default_info = {'python_version': 'py311'}
                self.step_8.set_python_version_info(default_info)
            
            # Run infrastructure deployment
            infrastructure_success = self.step_8.run_infrastructure_deployment()
            self.results[8] = infrastructure_success
            
            if infrastructure_success:
                self.logger.info("✅ Infrastructure deployment completed successfully!")
            else:
                self.logger.warning("⚠️ Infrastructure deployment failed or was cancelled")
            
            return pr_success and infrastructure_success
        else:
            self.logger.error("❌ PR creation failed, skipping infrastructure deployment")
            return False

    def extract_python_version_info(self):
        """Extract python version info from selected images using same logic as steps_3_4"""
        self.logger.info(f"🔍 DEBUG: self.selected_images = {self.selected_images}")
        if not self.selected_images:
            self.logger.warning("⚠️ No selected_images available for python version extraction")
            return {'python_version': 'py311'}
        try:
            import re
            sample_image = self.selected_images['training_cpu']
            self.logger.info(f"🔍 DEBUG: sample_image = {sample_image}")
            self.logger.info(f"🔍 DEBUG: sample_image.tag = {sample_image.tag}")
            python_match = re.search(r'-py(\d+)-', sample_image.tag)
            self.logger.info(f"🔍 DEBUG: python_match = {python_match}")
            if python_match:
                python_version = f"py{python_match.group(1)}"
            else:
                python_version = 'py311'  # fallback
            self.logger.info(f"✅ Extracted python version: {python_version}")
            result = {
                'python_version': python_version,
                'pytorch_version': self.selected_images.get('pytorch_version'),
                'cuda_version': self.selected_images.get('cuda_version')
            }
            self.logger.info(f"🔍 DEBUG: returning python_info = {result}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Could not extract python version info: {e}")
            return {'python_version': 'py311'}

    def step_8_infrastructure_deployment(self) -> bool:
        """Step 8: Infrastructure deployment after PR merge"""
        self.logger.info("Step 8: Infrastructure deployment")
        try:
            result = self.step_8.run_infrastructure_deployment()
            self.results[8] = result
            if result:
                self.logger.info("✅ Infrastructure deployment completed successfully")
            else:
                self.logger.warning("⚠️ Infrastructure deployment failed or was cancelled")
            return result
        except Exception as e:
            self.logger.error(f"❌ Step 8 infrastructure deployment failed: {e}")
            return False

    def run_infrastructure_only(self):
        """Run only the Infrastructure Deployment (Step 8)"""
        self.logger.info("🏗️ Running Infrastructure Deployment only...")
        try:
            result = self.step_8_infrastructure_deployment()
            self.results[8] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ Infrastructure deployment failed: {e}")
            return False
        
    def run_quick_checks_only(self):
        """Run only the Quick Checks Agent"""
        self.logger.info("🔍 Running Quick Checks Agent only...")
        try:
            result = self.quick_checks_agent.run_quick_checks_agent()
            self.results['quick_checks'] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ Quick Checks Agent failed: {e}")
            return False
        
    def step_7_create_pr(self) -> bool:
        """Step 7: Create Pull Request (legacy method - now redirects to complete workflow)"""
        return self.step_7_create_pr_with_complete_workflow()

    def run_automation_with_testing(self, steps_only=None, enable_pip_check=True, enable_autogluon_tests=True, enable_sagemaker_tests=True):
        """Run automation with integrated testing (security testing now automatic with PR creation)"""
        try:
            self.logger.info("🚀 Starting Enhanced AutoGluon release automation...")
            
            if self.should_run_steps([1, 2, 5], steps_only):
                self.logger.info("Running Steps 1, 2, and 5...")
                step_125_results = self.steps_125.run_steps(steps_only)
                self.results.update(step_125_results)
            
            if self.should_run_steps([3, 4], steps_only):
                self.logger.info("Running Steps 3 and 4...")
                step_34_results = self.steps_34.run_steps(steps_only)
                self.results.update(step_34_results)
                self.selected_images = self.steps_34.get_selected_images()
            
            if self.should_run_steps([6], steps_only):
                self.logger.info("Running Step 6...")
                step_6_results = self.step_6.run_steps(steps_only)
                self.results.update(step_6_results)
                
                if step_6_results.get(6, False):
                    if enable_pip_check:
                        self.logger.info("🤖 Running Pip Check Agent...")
                        pip_check_success = self.pip_check_agent.run_pip_check_agent()
                        self.results['pip_check'] = pip_check_success
                        if not pip_check_success:
                            self.logger.warning("⚠️ Pip check agent found issues that need manual review")
                    
                    if enable_autogluon_tests:
                        self.logger.info("🧪 Running AutoGluon Test Agent...")
                        autogluon_test_success = self.autogluon_test_agent.run_autogluon_test_agent()
                        self.results['autogluon_tests'] = autogluon_test_success
                        if not autogluon_test_success:
                            self.logger.warning("⚠️ AutoGluon tests found issues that need manual review")

                    if enable_sagemaker_tests:
                        self.logger.info("🧪 Running SageMaker Test Agent...")
                        sagemaker_test_success = self.sagemaker_test_agent.run_sagemaker_test_agent()
                        self.results['sagemaker_tests'] = sagemaker_test_success
                        if not sagemaker_test_success:
                            self.logger.warning("⚠️ SageMaker tests found issues that need manual review")
            
            self.print_enhanced_summary(steps_only)
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced automation failed: {e}")
            self.logger.exception("Full error details:")
            raise

    def run_complete_automation(self, steps_only=None, enable_pip_check=True, enable_autogluon_tests=True, enable_sagemaker_tests=True):
        """Run complete automation including Steps 1-6, then PR + Infrastructure + Release Images + Asimov Scan"""
        try:
            self.logger.info("🚀 Starting Complete AutoGluon release automation...")
            
            # First run the normal steps 1-6 with testing
            automation_results = self.run_automation_with_testing(
                steps_only=steps_only,
                enable_pip_check=enable_pip_check,
                enable_autogluon_tests=enable_autogluon_tests,
                enable_sagemaker_tests=enable_sagemaker_tests
            )
            
            # Check if steps 1-6 were successful
            core_steps_success = all(
                self.results.get(step, False) for step in [1, 2, 3, 4, 5, 6]
                if not steps_only or step in steps_only
            )
            
            if core_steps_success:
                self.logger.info("✅ Core steps (1-6) completed successfully, proceeding with complete workflow...")
                
                # Run the complete workflow (PR + Infrastructure + Release Images + Asimov Scan)
                complete_workflow_success = self.step_7_create_pr_with_complete_workflow()
                
                if complete_workflow_success:
                    self.logger.info("✅ Complete automation workflow finished successfully!")
                else:
                    self.logger.warning("⚠️ Complete workflow had some failures")
                    
                self.print_enhanced_summary(steps_only)
                return self.results
            else:
                self.logger.error("❌ Core steps failed, skipping complete workflow")
                self.print_enhanced_summary(steps_only)
                return self.results
                
        except Exception as e:
            self.logger.error(f"❌ Complete automation failed: {e}")
            self.logger.exception("Full error details:")
            raise

    def should_run_steps(self, step_numbers, steps_only):
        """Check if any of the given step numbers should be run"""
        if not steps_only:
            return True
        return any(step in steps_only for step in step_numbers)

    def print_enhanced_summary(self, steps_only):
        """Print enhanced automation summary with agentic details"""
        print("\n" + "="*70)
        print("🤖 COMPLETE AUTOGLUON RELEASE AUTOMATION SUMMARY")
        print("="*70)
        
        if steps_only:
            completed_steps = [
                str(step) for step in sorted(steps_only) 
                if self.results.get(step, False)
            ]
            failed_steps = [
                str(step) for step in sorted(steps_only) 
                if step in self.results and not self.results[step]
            ]
        else:
            successful_steps = [
                str(k) for k, v in self.results.items() 
                if v and k not in ['pip_check', 'autogluon_tests', 'sagemaker_tests', 'security_tests', 'quick_checks', 'release_images', 'asimov_scan']
            ]
            failed_steps = [
                str(k) for k, v in self.results.items() 
                if not v and k not in ['pip_check', 'autogluon_tests', 'sagemaker_tests', 'security_tests', 'quick_checks', 'release_images', 'asimov_scan']
            ]
            completed_steps = successful_steps
        
        if completed_steps:
            print(f"✅ Completed steps: {', '.join(completed_steps)}")
        if failed_steps:
            print(f"❌ Failed steps: {', '.join(failed_steps)}")
        
        # Agent results
        if 'pip_check' in self.results:
            if self.results['pip_check']:
                print("✅ Pip Check Agent: All dependency checks passed")
            else:
                print("⚠️ Pip Check Agent: Issues found (check logs for details)")
                
        if 'autogluon_tests' in self.results:
            if self.results['autogluon_tests']:
                print("✅ AutoGluon Test Agent: All tests passed")
            else:
                print("⚠️ AutoGluon Test Agent: Test failures found (check logs for details)")
                
        if 'sagemaker_tests' in self.results:
            if self.results['sagemaker_tests']:
                print("✅ SageMaker Test Agent: All tests passed")
            else:
                print("⚠️ SageMaker Test Agent: Test failures found (check logs for details)")
        
        if 'security_tests' in self.results:
            if self.results['security_tests']:
                print("🤖 Agentic Security Analysis: ✅ All security issues resolved autonomously")
            else:
                print("🤖 Agentic Security Analysis: ⚠️ Some issues found (AI provided recommendations)")
        
        if 'quick_checks' in self.results:
            if self.results['quick_checks']:
                print("✅ Quick Checks Agent: dlc-pr-quick-checks handled successfully")
            else:
                print("⚠️ Quick Checks Agent: Issues found")
        
        if 7 in self.results:
            if self.results[7]:
                print("✅ Pull Request: Successfully created with agentic security analysis")
            else:
                print("⚠️ Pull Request: Creation failed")
        
        if 8 in self.results:
            if self.results[8]:
                print("🏗️ Infrastructure Deployment: ✅ Successfully completed")
            else:
                print("🏗️ Infrastructure Deployment: ⚠️ Failed or cancelled")
        
        if 'release_images' in self.results:
            if self.results['release_images']:
                print("📦 AutoGluon Release Images: ✅ Successfully completed")
            else:
                print("📦 AutoGluon Release Images: ⚠️ Failed or had issues")
                
        if 'asimov_scan' in self.results:
            if self.results['asimov_scan']:
                print("🔒 Asimov Security Scan: ✅ Successfully completed")
            else:
                print("🔒 Asimov Security Scan: ⚠️ Failed or had issues")
        
        print(f"\n📋 Release Information:")
        print(f"   Version: {self.current_version}")
        print(f"   Type: {'Major' if self.is_major_release else 'Minor'}")
        print(f"   Branch: autogluon-{self.current_version}-release")
        print(f"   🤖 AI Enhancement: Autonomous security vulnerability analysis and fixing")
        
        if 8 in self.results:
            print(f"   🏗️ Infrastructure: Brazil workspace setup and config deployment")
            
        if 'release_images' in self.results:
            print(f"   📦 Release Images: YAML files updated and available_images.md updated")
            
        if 'asimov_scan' in self.results:
            print(f"   🔒 Security Scan: Asimov workspace setup and images.py updated")
        
        if self.selected_images:
            print(f"   PyTorch: {self.selected_images.get('pytorch_version', 'N/A')}")
            print(f"   CUDA: {self.selected_images.get('cuda_version', 'N/A')}")
        
        print("="*70)

    # Other methods remain the same...
    def run_automation(self, steps_only=None):
        """Backward compatible method (security testing now automatic with PR creation)"""
        return self.run_automation_with_testing(
            steps_only, 
            enable_pip_check=False, 
            enable_autogluon_tests=False,
            enable_sagemaker_tests=False
        )

    def run_steps_1_2_5(self, steps_only=None):
        """Run only steps 1, 2, and 5"""
        valid_steps = [s for s in (steps_only or [1, 2, 5]) if s in [1, 2, 5]]
        return self.steps_125.run_steps(valid_steps)

    def run_steps_3_4(self, steps_only=None):
        """Run only steps 3 and 4"""
        valid_steps = [s for s in (steps_only or [3, 4]) if s in [3, 4]]
        results = self.steps_34.run_steps(valid_steps)
        self.selected_images = self.steps_34.get_selected_images()
        return results

    def run_step_6(self):
        """Run only step 6"""
        return self.step_6.run_steps([6])

    def get_selected_images(self):
        """Get selected images from steps 3-4"""
        return self.selected_images


def main():
    parser = argparse.ArgumentParser(description='Complete AutoGluon DLC Release Automation with Full Workflow')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)') 
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    
    # Individual component options
    parser.add_argument('--steps-125', action='store_true', help='Run steps 1, 2, and 5')
    parser.add_argument('--steps-34', action='store_true', help='Run steps 3 and 4')
    parser.add_argument('--steps-both', action='store_true', help='Run both: steps 1,2,5 and 3,4')
    parser.add_argument('--step-6', action='store_true', help='Run all steps 1-6')
    parser.add_argument('--step-6-only', action='store_true', help='Run only step 6')
    parser.add_argument('--pip-check', action='store_true', help='Run pip check only')
    parser.add_argument('--autogluon-tests', action='store_true', help='Run AutoGluon tests only')
    parser.add_argument('--sagemaker', action='store_true', help='Run SageMaker tests only')
    parser.add_argument('--infrastructure', action='store_true', help='Run infrastructure deployment (Step 8) only')
    parser.add_argument('--quick-checks', action='store_true', help='Run Quick Checks Agent only')
    parser.add_argument('--release-images', action='store_true', help='Run AutoGluon Release Images automation only')
    parser.add_argument('--asimov-scan', action='store_true', help='Run Asimov Security Scan automation only')
    parser.add_argument('--post-pr-workflow', action='store_true', help='Run post-PR workflow: Infrastructure + Release Images + Asimov Scan only')
    
    # Workflow options
    parser.add_argument('--create-pr', action='store_true', help='Create PR with security analysis and quick checks')
    parser.add_argument('--pr-only', action='store_true', help='Create PR with complete workflow (Infrastructure + Release Images + Asimov Scan)')
    parser.add_argument('--complete', action='store_true', help='Run complete automation workflow (Steps 1-6 + PR + Infrastructure + Release Images + Asimov Scan)')
    parser.add_argument('--agentic-security', action='store_true', help='Run agentic security analysis only')
    parser.add_argument('--agentic-full', action='store_true', help='Run full agentic testing suite')
    
    args = parser.parse_args()
    
    automation = EnhancedAutoGluonReleaseAutomation(
        args.current_version,
        args.previous_version, 
        args.fork_url
    )
    
    success = False
    
    if args.complete:
        print("🚀 Running Complete AutoGluon Release Automation...")
        results = automation.run_complete_automation(
            enable_pip_check=True, 
            enable_autogluon_tests=True,
            enable_sagemaker_tests=True
        )
        success = all(results.values())
    elif args.post_pr_workflow:
        print("🚀 Running Post-PR Workflow: Infrastructure → Release Images → Asimov Scan...")
        success = automation.run_post_pr_workflow_only()
        if success:
            print("✅ Post-PR workflow completed successfully!")
        else:
            print("❌ Post-PR workflow failed")
    elif args.asimov_scan:
        print("🔒 Running Asimov Security Scan Automation only...")
        success = automation.run_asimov_scan_only()
    elif args.release_images:
        print("📦 Running AutoGluon Release Images Automation only...")
        success = automation.run_release_images_only()
    elif args.quick_checks:
        print("🔍 Running Quick Checks Agent only...")
        success = automation.run_quick_checks_only()
    elif args.infrastructure:
        print("🏗️ Running Infrastructure Deployment (Step 8) only...")
        success = automation.run_infrastructure_only()
    elif args.autogluon_tests:
        print("🧪 Running AutoGluon Test Agent only...")
        success = automation.run_autogluon_tests_only()
    elif args.pr_only:
        print("🚀 Creating Pull Request with complete workflow...")
        success = automation.step_7_create_pr_with_complete_workflow()
        if success:
            print("✅ Pull Request and complete workflow completed!")
        else:
            print("❌ Pull Request or workflow steps failed")
    elif args.create_pr:
        print("🚀 Creating Pull Request with security analysis...")
        success = automation.step_7_create_pr_with_security()
        if success:
            print("✅ Pull Request and security analysis completed!")
        else:
            print("❌ Pull Request creation or security analysis failed")
    elif args.steps_125:
        print("🔧 Running steps 1, 2, and 5...")
        results = automation.run_steps_1_2_5()
        success = all(results.values())
    elif args.steps_34:
        print("🔧 Running steps 3 and 4...")
        results = automation.run_steps_3_4()
        success = all(results.values())
    elif args.steps_both:
        print("🔧 Running steps 1,2,5 and 3,4...")
        results_125 = automation.run_steps_1_2_5()
        results_34 = automation.run_steps_3_4()
        success = all(results_125.values()) and all(results_34.values())
    elif args.step_6_only:
        print("🔧 Running only step 6...")
        results = automation.run_step_6()
        success = all(results.values())
    else:
        # Default: run complete automation
        print("🔧 Running complete automation workflow (default)...")
        results = automation.run_complete_automation(
            enable_pip_check=True, 
            enable_autogluon_tests=True,
            enable_sagemaker_tests=True
        )
        success = all(results.values())
    
    automation.print_enhanced_summary(None)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()