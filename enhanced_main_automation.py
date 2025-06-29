import argparse
import logging
from steps_1_2_5 import Steps125Automation
from steps_3_4 import Steps34Automation  
from step_6 import Step6Automation
from pip_check_agent import PipCheckAgent
from sagemaker_test_agent import SageMakerTestAgent
from security_test_agent import SecurityTestAgent
from github_pr_automation import GitHubPRAutomation
from quick_checks_agent import QuickChecksAgent

class EnhancedAutoGluonReleaseAutomation:
    """Enhanced orchestrator with agentic pip check, SageMaker testing, and integrated PR + security capabilities"""
    
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
        self.pip_check_agent = PipCheckAgent(current_version, previous_version, fork_url)
        self.sagemaker_test_agent = SageMakerTestAgent(current_version, previous_version, fork_url)
        self.security_test_agent = SecurityTestAgent(current_version, previous_version, fork_url)
        self.quick_checks_agent=QuickChecksAgent(current_version,previous_version,fork_url)

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
                    quick_checks_success=self.quick_checks_agent.run_quick_checks_agent()
                    self.results['quick_checks']=quick_checks_success

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
        """Step 7: Create Pull Request (legacy method - now redirects to integrated version)"""
        return self.step_7_create_pr_with_security()

    def run_automation_with_testing(self, steps_only=None, enable_pip_check=True, enable_sagemaker_tests=True):
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

    def run_automation_with_pr(self, steps_only=None, enable_pip_check=True, enable_sagemaker_tests=True, create_pr=True):
        """Run automation with integrated testing and automatic PR + security testing"""
        try:
            results = self.run_automation_with_testing(steps_only, enable_pip_check, enable_sagemaker_tests, enable_security_tests=False)
            # Check if we should create PR
            if create_pr and self.should_run_steps([7], steps_only):
                # Only create PR if core steps completed successfully
                core_steps_success = all(
                    v for k, v in results.items() 
                    if k not in ['pip_check', 'sagemaker_tests', 'security_tests']
                )
                
                if core_steps_success:
                    self.logger.info("🚀 Core steps completed successfully, creating PR with automatic security testing...")
                    pr_and_security_success = self.step_7_create_pr_with_security()
                    results[7] = pr_and_security_success
                    results['security_tests'] = self.results.get('security_tests', False)
                    
                    if pr_and_security_success:
                        self.logger.info("✅ Pull Request created and security tests completed!")
                    else:
                        self.logger.warning("⚠️ Pull Request creation or security testing had issues")
                else:
                    self.logger.warning("⚠️ Skipping PR creation due to failed core steps")
                    results[7] = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced automation with PR failed: {e}")
            self.logger.exception("Full error details:")
            raise
    def run_agentic_security_only(self):
        """Run only the Agentic Security Test Agent"""
        self.logger.info("🤖 Running Agentic Security Test Agent...")
        try:
            # First check if PR exists
            pr_number = self.security_test_agent.get_current_pr_number()
            
            if not pr_number:
                self.logger.warning("⚠️ No existing PR found for security testing")
                self.logger.info("🚀 Creating PR first (required for security testing)...")
                
                # Create PR first since security testing requires it
                pr_success = self.pr_automation.create_pull_request()
                if not pr_success:
                    self.logger.error("❌ Failed to create PR - cannot run agentic security tests")
                    return False
                
                self.logger.info("✅ PR created successfully")
                
                # Get the new PR number
                pr_number = self.security_test_agent.get_current_pr_number()
                
                if pr_number:
                    # Wait for security tests to complete before AI analysis
                    self.logger.info("⏳ Waiting for security tests to complete...")
                    self.security_test_agent.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=45)
                
                # Mark PR creation as successful
                self.results[7] = True
            
            # Run agentic security analysis and fixing
            self.logger.info("🤖 Running autonomous AI security analysis...")
            result = self.security_test_agent.run_agentic_security_analysis()
            self.results['security_tests'] = result
            
            return result
        except Exception as e:
            self.logger.error(f"❌ Agentic security test agent failed: {e}")
            return False

    def run_full_agentic_testing_suite(self, include_security=True):
        """Run all testing agents with agentic security analysis"""
        self.logger.info("🔬 Running Full Agentic Testing Suite...")
        try:
            pip_success = self.run_pip_check_only()
            sagemaker_success = self.run_sagemaker_tests_only()
            
            security_success = True
            if include_security:
                security_success = self.run_agentic_security_only()
            
            overall_success = pip_success and sagemaker_success and security_success
            
            self.logger.info("📊 Agentic Testing Suite Results:")
            self.logger.info(f"   Pip Check: {'✅ PASSED' if pip_success else '❌ FAILED'}")
            self.logger.info(f"   SageMaker Tests: {'✅ PASSED' if sagemaker_success else '❌ FAILED'}")
            if include_security:
                self.logger.info(f"   🤖 Agentic Security: {'✅ PASSED' if security_success else '❌ FAILED'}")
            self.logger.info(f"   Overall: {'✅ PASSED' if overall_success else '❌ FAILED'}")
            return overall_success
        except Exception as e:
            self.logger.error(f"❌ Full agentic testing suite failed: {e}")
            return False

    def run_automation_with_agentic_pr(self, steps_only=None, enable_pip_check=True, enable_sagemaker_tests=True, create_pr=True):
        """Run automation with agentic PR creation and security testing"""
        try:
            # Run the main automation steps
            results = self.run_automation_with_testing(steps_only, enable_pip_check, enable_sagemaker_tests)
            
            # Check if we should create PR with agentic security
            if create_pr and self.should_run_steps([7], steps_only):
                # Only create PR if core steps completed successfully
                core_steps_success = all(
                    v for k, v in results.items() 
                    if k not in ['pip_check', 'sagemaker_tests', 'security_tests']
                )
                
                if core_steps_success:
                    self.logger.info("🚀 Core steps completed successfully, creating PR with agentic security analysis...")
                    pr_and_security_success = self.step_7_create_pr_with_security()
                    results[7] = pr_and_security_success
                    results['security_tests'] = self.results.get('security_tests', False)
                    
                    if pr_and_security_success:
                        self.logger.info("✅ Pull Request created and agentic security analysis completed!")
                    else:
                        self.logger.warning("⚠️ Pull Request creation or agentic security analysis had issues")
                else:
                    self.logger.warning("⚠️ Skipping PR creation due to failed core steps")
                    results[7] = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced automation with agentic PR failed: {e}")
            self.logger.exception("Full error details:")
            raise

    def run_pip_check_only(self):
        """Run only the pip check agent"""
        self.logger.info("🤖 Running Pip Check Agent only...")
        try:
            result = self.pip_check_agent.run_pip_check_agent()
            self.results['pip_check'] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ Pip check agent failed: {e}")
            return False

    def run_sagemaker_tests_only(self):
        """Run only the SageMaker test agent"""
        self.logger.info("🧪 Running SageMaker Test Agent only...")
        try:
            result = self.sagemaker_test_agent.run_sagemaker_test_agent()
            self.results['sagemaker_tests'] = result
            return result
        except Exception as e:
            self.logger.error(f"❌ SageMaker test agent failed: {e}")
            return False

    def run_security_tests_only(self):
        """Run only the Agentic Security Test Agent (ensures PR exists first and waits for tests to complete)"""
        self.logger.info("🤖 Running Agentic Security Test Agent...")
        try:
            # First check if PR exists
            pr_number = self.security_test_agent.get_current_pr_number()
            
            if not pr_number:
                self.logger.warning("⚠️ No existing PR found for security testing")
                self.logger.info("🚀 Creating PR first (required for security testing)...")
                pr_success = self.pr_automation.create_pull_request()
                if not pr_success:
                    self.logger.error("❌ Failed to create PR - cannot run agentic security tests")
                    return False
                
                self.logger.info("✅ PR created successfully")
                
                # Get the new PR number
                pr_number = self.security_test_agent.get_current_pr_number()
                
                if pr_number:
                    # Wait for security tests to complete before analyzing
                    self.logger.info("⏳ Waiting for security tests to complete...")
                    self.security_test_agent.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=45)
                
                # Mark PR creation as successful
                self.results[7] = True
            
            # Run agentic security analysis and fixing
            self.logger.info("🤖 Running autonomous AI security analysis...")
            result = self.security_test_agent.run_agentic_security_analysis()
            self.results['security_tests'] = result
            
            return result
        except Exception as e:
            self.logger.error(f"❌ Agentic security test agent failed: {e}")
            return False

    def should_run_steps(self, step_numbers, steps_only):
        """Check if any of the given step numbers should be run"""
        if not steps_only:
            return True
        return any(step in steps_only for step in step_numbers)

    def print_enhanced_summary(self, steps_only):
        """Print enhanced automation summary with agentic details"""
        print("\n" + "="*70)
        print("🤖 AGENTIC AUTOGLUON RELEASE AUTOMATION SUMMARY")
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
                if v and k not in ['pip_check', 'sagemaker_tests', 'security_tests']
            ]
            failed_steps = [
                str(k) for k, v in self.results.items() 
                if not v and k not in ['pip_check', 'sagemaker_tests', 'security_tests']
            ]
            completed_steps = successful_steps
        
        if completed_steps:
            print(f"✅ Completed steps: {', '.join(completed_steps)}")
        if failed_steps:
            print(f"❌ Failed steps: {', '.join(failed_steps)}")
        
        if 'pip_check' in self.results:
            if self.results['pip_check']:
                print("✅ Pip Check Agent: All dependency checks passed")
            else:
                print("⚠️ Pip Check Agent: Issues found (check logs for details)")
        
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
        
        if 7 in self.results:
            if self.results[7]:
                print("✅ Pull Request: Successfully created with agentic security analysis")
            else:
                print("⚠️ Pull Request: Creation failed")
        if 'quick_checks' in self.results:
            if self.results['quick_checks']:
                print("Quick checks agent: dlc-pr-quick-checks handled successfully")
            else:
                print("Quick checks agent: Issues found")
        print(f"\n📋 Release Information:")
        print(f"   Version: {self.current_version}")
        print(f"   Type: {'Major' if self.is_major_release else 'Minor'}")
        print(f"   Branch: autogluon-{self.current_version}-release")
        print(f"   🤖 AI Enhancement: Autonomous security vulnerability analysis and fixing")
        
        if self.selected_images:
            print(f"   PyTorch: {self.selected_images.get('pytorch_version', 'N/A')}")
            print(f"   CUDA: {self.selected_images.get('cuda_version', 'N/A')}")
        
        print("="*70)

    # Backward compatible methods
    def run_automation(self, steps_only=None):
        """Backward compatible method (security testing now automatic with PR creation)"""
        return self.run_automation_with_testing(
            steps_only, 
            enable_pip_check=False, 
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

    def monitor_pr_tests(self, pr_number: int = None, show_pending: bool = True):
        """Monitor test status for the current or specified PR"""
        return self.pr_automation.monitor_pr_status(pr_number, show_pending)

    def wait_for_pr_tests(self, pr_number: int = None, max_wait_minutes: int = 120):
        """Wait for PR tests to complete"""
        return self.pr_automation.wait_for_tests(pr_number, max_wait_minutes)

    def run_automation_with_pr_monitoring(self, steps_only=None, enable_pip_check=True, enable_sagemaker_tests=True, create_pr=True, wait_for_tests=False, max_wait_minutes=60):
        """Run automation with PR creation"""
        try:
            # Run the main automation steps
            results = self.run_automation_with_testing(steps_only, enable_pip_check, enable_sagemaker_tests, enable_security_tests=False)
            
            # Check if we should create PR
            if create_pr and self.should_run_steps([7], steps_only):
                # Only create PR if core steps completed successfully
                core_steps_success = all(
                    v for k, v in results.items() 
                    if k not in ['pip_check', 'sagemaker_tests', 'security_tests']
                )
                
                if core_steps_success:
                    self.logger.info("🚀 Core steps completed successfully, creating PR with automatic security testing...")
                    pr_and_security_success = self.step_7_create_pr_with_security()
                    results[7] = pr_and_security_success
                    results['security_tests'] = self.results.get('security_tests', False)
                    
                    if pr_and_security_success:
                        self.logger.info("✅ Pull Request created and security tests completed!")
                        
                        # Monitor tests if requested
                        if wait_for_tests:
                            self.logger.info("⏳ Waiting for PR tests to complete...")
                            tests_passed = self.wait_for_pr_tests(max_wait_minutes=max_wait_minutes)
                            results['tests_passed'] = tests_passed
                            
                            if tests_passed:
                                self.logger.info("✅ All PR tests passed!")
                            else:
                                self.logger.warning("⚠️ Some PR tests failed or timed out")
                    else:
                        self.logger.warning("⚠️ Pull Request creation or security testing failed")
                else:
                    self.logger.warning("⚠️ Skipping PR creation due to failed core steps")
                    results[7] = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced automation with PR monitoring failed: {e}")
            self.logger.exception("Full error details:")
            raise

    def get_selected_images(self):
        """Get selected images from steps 3-4"""
        return self.selected_images


def main():
    parser = argparse.ArgumentParser(description='Agentic AutoGluon DLC Release Automation with AI Security Analysis')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)') 
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    
    # Existing options
    parser.add_argument('--steps-125', action='store_true', help='Run steps 1, 2, and 5')
    parser.add_argument('--steps-34', action='store_true', help='Run steps 3 and 4')
    parser.add_argument('--steps-both', action='store_true', help='Run both: steps 1,2,5 and 3,4')
    parser.add_argument('--step-6', action='store_true', help='Run all steps 1-6')
    parser.add_argument('--step-6-only', action='store_true', help='Run only step 6')
    parser.add_argument('--pip-check', action='store_true', help='Run pip check (or all steps + pip check)')
    parser.add_argument('--sagemaker', action='store_true', help='Run SageMaker tests (or all steps + pip check + sagemaker)')
    
    # Updated agentic options
    parser.add_argument('--create-pr', action='store_true', help='Create PR after successful completion (with agentic security)')
    parser.add_argument('--pr-only', action='store_true', help='Create PR with agentic security analysis')
    parser.add_argument('--agentic-security', action='store_true', help='Run agentic security analysis (creates PR if needed)')
    parser.add_argument('--agentic-full', action='store_true', help='Run full agentic testing suite')
    
    # Monitoring options
    parser.add_argument('--monitor-pr', type=int, help='Monitor test status for specific PR number')
    parser.add_argument('--max-wait', type=int, default=120, help='Maximum minutes to wait for tests (default: 120)')
    parser.add_argument('--quick-checks',action='store_true',help='Run Quick Checks Agent only')
    args = parser.parse_args()
    
    automation = EnhancedAutoGluonReleaseAutomation(
        args.current_version,
        args.previous_version, 
        args.fork_url
    )
    
    if args.quick_checks:
        print("Running quick checks agent...")
        success = automation.run_quick_checks_only()
    elif args.agentic_security:
        print("🤖 Running Agentic Security Analysis...")
        success = automation.run_agentic_security_only()
    elif args.agentic_full:
        print("🤖 Running Full Agentic Testing Suite...")
        success = automation.run_full_agentic_testing_suite()
    elif args.pr_only:
        print("🚀 Creating Pull Request with agentic security analysis...")
        success = automation.step_7_create_pr_with_security()
        if success:
            print("✅ Pull Request created and agentic security analysis completed!")
        else:
            print("❌ Pull Request creation or agentic security analysis failed")
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
    elif args.step_6:
        print("🔧 Running all steps 1-6...")
        results = automation.run_automation_with_agentic_pr(
            enable_pip_check=False, 
            enable_sagemaker_tests=False,
            create_pr=args.create_pr
        )
        success = all(v for k, v in results.items() if k not in ['pip_check', 'sagemaker_tests', 'security_tests'])
    elif args.pip_check:
        print("🔧 Running all steps 1-6 + pip check...")
        results = automation.run_automation_with_agentic_pr(
            enable_pip_check=True, 
            enable_sagemaker_tests=False,
            create_pr=args.create_pr
        )
        success = all(v for k, v in results.items() if k not in ['sagemaker_tests'])
    elif args.sagemaker:
        print("🔧 Running all steps 1-6 + pip check + SageMaker tests...")
        results = automation.run_automation_with_agentic_pr(
            enable_pip_check=True, 
            enable_sagemaker_tests=True,
            create_pr=args.create_pr
        )
        success = all(results.values())
    else:
        # Default full pipeline now includes PR creation automatically
        print("🔧 Running full agentic pipeline with PR creation...")
        results = automation.run_automation_with_agentic_pr(
            enable_pip_check=True, 
            enable_sagemaker_tests=True,
            create_pr=True  # Changed from args.create_pr to True
        )
        success = all(results.values())
    automation.print_enhanced_summary(None)
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()