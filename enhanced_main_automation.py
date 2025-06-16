import argparse
import logging
from steps_1_2_5 import Steps125Automation
from steps_3_4 import Steps34Automation  
from step_6 import Step6Automation
from pip_check_agent import PipCheckAgent
from sagemaker_test_agent import SageMakerTestAgent

class EnhancedAutoGluonReleaseAutomation:
    """Enhanced orchestrator with agentic pip check and SageMaker testing capabilities"""
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        self.current_version=current_version
        self.previous_version=previous_version
        self.fork_url=fork_url
        current_parts=current_version.split('.')
        previous_parts=previous_version.split('.')
        self.is_major_release=(current_parts[0] != previous_parts[0] or 
                               (current_parts[1] != previous_parts[1] and current_parts[2] == '0'))
        logging.basicConfig(level=logging.INFO)
        self.logger=logging.getLogger(__name__)
        self.logger.info(f"Enhanced AutoGluon Release Automation - Version {current_version}")
        self.logger.info(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
        self.steps_125=Steps125Automation(current_version, previous_version, fork_url)
        self.steps_34=Steps34Automation(current_version, previous_version, fork_url)
        self.step_6=Step6Automation(current_version, previous_version, fork_url)
        self.pip_check_agent=PipCheckAgent(current_version, previous_version, fork_url)
        self.sagemaker_test_agent=SageMakerTestAgent(current_version, previous_version, fork_url)
        self.results={}
        self.selected_images=None

    def run_automation_with_testing(self, steps_only=None, enable_pip_check=True, enable_sagemaker_tests=True):
        """Run automation with integrated testing"""
        try:
            self.logger.info("🚀 Starting Enhanced AutoGluon release automation...")
            if self.should_run_steps([1, 2, 5], steps_only):
                self.logger.info("Running Steps 1, 2, and 5...")
                step_125_results=self.steps_125.run_steps(steps_only)
                self.results.update(step_125_results)
            if self.should_run_steps([3, 4], steps_only):
                self.logger.info("Running Steps 3 and 4...")
                step_34_results=self.steps_34.run_steps(steps_only)
                self.results.update(step_34_results)
                self.selected_images=self.steps_34.get_selected_images()
            if self.should_run_steps([6], steps_only):
                self.logger.info("Running Step 6...")
                step_6_results=self.step_6.run_steps(steps_only)
                self.results.update(step_6_results)
                if step_6_results.get(6, False):
                    if enable_pip_check:
                        self.logger.info("🤖 Running Pip Check Agent...")
                        pip_check_success=self.pip_check_agent.run_pip_check_agent()
                        self.results['pip_check']=pip_check_success
                        if not pip_check_success:
                            self.logger.warning("⚠️ Pip check agent found issues that need manual review")
                    if enable_sagemaker_tests:
                        self.logger.info("🧪 Running SageMaker Test Agent...")
                        sagemaker_test_success=self.sagemaker_test_agent.run_sagemaker_test_agent()
                        self.results['sagemaker_tests']=sagemaker_test_success
                        if not sagemaker_test_success:
                            self.logger.warning("⚠️ SageMaker tests found issues that need manual review")
            self.print_enhanced_summary(steps_only)
            return self.results
        except Exception as e:
            self.logger.error(f"❌ Enhanced automation failed: {e}")
            self.logger.exception("Full error details:")
            raise

    def run_pip_check_only(self):
        """Run only the pip check agent"""
        self.logger.info("🤖 Running Pip Check Agent only...")
        try:
            result=self.pip_check_agent.run_pip_check_agent()
            self.results['pip_check']=result
            return result
        except Exception as e:
            self.logger.error(f"❌ Pip check agent failed: {e}")
            return False

    def run_sagemaker_tests_only(self):
        """Run only the SageMaker test agent"""
        self.logger.info("🧪 Running SageMaker Test Agent only...")
        try:
            result=self.sagemaker_test_agent.run_sagemaker_test_agent()
            self.results['sagemaker_tests']=result
            return result
        except Exception as e:
            self.logger.error(f"❌ SageMaker test agent failed: {e}")
            return False

    def run_full_testing_suite(self):
        """Run both pip check and SageMaker tests"""
        self.logger.info("🔬 Running Full Testing Suite...")
        try:
            pip_success=self.run_pip_check_only()
            sagemaker_success=self.run_sagemaker_tests_only()
            overall_success=pip_success and sagemaker_success
            self.logger.info("📊 Testing Suite Results:")
            self.logger.info(f"   Pip Check: {'✅ PASSED' if pip_success else '❌ FAILED'}")
            self.logger.info(f"   SageMaker Tests: {'✅ PASSED' if sagemaker_success else '❌ FAILED'}")
            self.logger.info(f"   Overall: {'✅ PASSED' if overall_success else '❌ FAILED'}")
            return overall_success
        except Exception as e:
            self.logger.error(f"❌ Full testing suite failed: {e}")
            return False

    def run_iterative_fix_cycle(self, max_iterations=3, enable_sagemaker_tests=True):
        """Run iterative build-test-fix cycle with optional SageMaker testing"""
        self.logger.info("🔄 Starting iterative fix cycle...")
        for iteration in range(max_iterations):
            self.logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            self.logger.info("🏗️ Building images...")
            build_success=self.step_6.step6_build_upload_docker()
            if not build_success:
                self.logger.error(f"❌ Build failed on iteration {iteration + 1}")
                return False
            self.logger.info("🧪 Running pip check...")
            pip_check_success=self.pip_check_agent.run_pip_check_agent()
            sagemaker_success=True
            if enable_sagemaker_tests:
                self.logger.info("🧪 Running SageMaker tests...")
                sagemaker_success=self.sagemaker_test_agent.run_sagemaker_test_agent()
            if pip_check_success and sagemaker_success:
                self.logger.info(f"✅ All checks passed on iteration {iteration + 1}")
                return True
            else:
                test_results=[]
                if not pip_check_success:
                    test_results.append("pip check issues")
                if not sagemaker_success:
                    test_results.append("SageMaker test failures")
                self.logger.info(f"ℹ️ Issues found on iteration {iteration + 1}: {', '.join(test_results)}")
                if iteration == max_iterations - 1:
                    self.logger.info("ℹ️ Note: If automatic fixes were applied, manual review may resolve remaining issues")
        self.logger.warning(f"⚠️ Completed {max_iterations} iterations - check logs for final status")
        return True  

    def should_run_steps(self, step_numbers, steps_only):
        """Check if any of the given step numbers should be run"""
        if not steps_only:
            return True
        return any(step in steps_only for step in step_numbers)

    def print_enhanced_summary(self, steps_only):
        """Print enhanced automation summary"""
        print("\n" + "="*70)
        print("🤖 ENHANCED AUTOGLUON RELEASE AUTOMATION SUMMARY")
        print("="*70)
        if steps_only:
            completed_steps=[str(step) for step in sorted(steps_only) if self.results.get(step, False)]
            failed_steps=[str(step) for step in sorted(steps_only) if step in self.results and not self.results[step]]
        else:
            successful_steps=[str(k) for k, v in self.results.items() if v and k not in ['pip_check', 'sagemaker_tests']]
            failed_steps=[str(k) for k, v in self.results.items() if not v and k not in ['pip_check', 'sagemaker_tests']]
            completed_steps=successful_steps
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
        print(f"\n📋 Release Information:")
        print(f"   Version: {self.current_version}")
        print(f"   Type: {'Major' if self.is_major_release else 'Minor'}")
        print(f"   Branch: autogluon-{self.current_version}-release")
        if self.selected_images:
            print(f"   PyTorch: {self.selected_images.get('pytorch_version', 'N/A')}")
            print(f"   CUDA: {self.selected_images.get('cuda_version', 'N/A')}")
        print("="*70)

    def run_automation(self, steps_only=None):
        """Backward compatible method"""
        return self.run_automation_with_testing(steps_only, enable_pip_check=False, enable_sagemaker_tests=False)

    def run_steps_1_2_5(self, steps_only=None):
        """Run only steps 1, 2, and 5"""
        valid_steps=[s for s in (steps_only or [1, 2, 5]) if s in [1, 2, 5]]
        return self.steps_125.run_steps(valid_steps)

    def run_steps_3_4(self, steps_only=None):
        """Run only steps 3 and 4"""
        valid_steps=[s for s in (steps_only or [3, 4]) if s in [3, 4]]
        results=self.steps_34.run_steps(valid_steps)
        self.selected_images=self.steps_34.get_selected_images()
        return results

    def run_step_6(self):
        """Run only step 6"""
        return self.step_6.run_steps([6])

    def get_selected_images(self):
        """Get selected images from steps 3-4"""
        return self.selected_images

def main():
    parser=argparse.ArgumentParser(description='Enhanced AutoGluon DLC Release Automation')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)') 
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    parser.add_argument('--steps-125', action='store_true', help='Run steps 1, 2, and 5')
    parser.add_argument('--steps-34', action='store_true', help='Run steps 3 and 4')
    parser.add_argument('--steps-both', action='store_true', help='Run both: steps 1,2,5 and 3,4')
    parser.add_argument('--step-6', action='store_true', help='Run step 6 (or all steps 1-6)')
    parser.add_argument('--pip-check', action='store_true', help='Run pip check (or all steps + pip check)')
    parser.add_argument('--sagemaker', action='store_true', help='Run SageMaker tests (or all steps + pip check + sagemaker)')
    args=parser.parse_args()
    automation=EnhancedAutoGluonReleaseAutomation(
        args.current_version,
        args.previous_version, 
        args.fork_url
    )
    if args.steps_125:
        print("🔧 Running steps 1, 2, and 5...")
        results=automation.run_steps_1_2_5()
        success=all(results.values())
    elif args.steps_34:
        print("🔧 Running steps 3 and 4...")
        results=automation.run_steps_3_4()
        success=all(results.values())
    elif args.steps_both:
        print("🔧 Running steps 1,2,5 and 3,4...")
        results_125=automation.run_steps_1_2_5()
        results_34=automation.run_steps_3_4()
        success=all(results_125.values()) and all(results_34.values())
    elif args.step_6:
        print("🔧 Running all steps 1-6...")
        results=automation.run_automation_with_testing(enable_pip_check=False, enable_sagemaker_tests=False)
        success=all(v for k, v in results.items() if k not in ['pip_check', 'sagemaker_tests'])
    elif args.pip_check:
        print("🔧 Running all steps 1-6 + pip check...")
        results=automation.run_automation_with_testing(enable_pip_check=True, enable_sagemaker_tests=False)
        success=all(v for k, v in results.items() if k != 'sagemaker_tests')
    elif args.sagemaker:
        print("🔧 Running all steps 1-6 + pip check + SageMaker tests...")
        results=automation.run_automation_with_testing(enable_pip_check=True, enable_sagemaker_tests=True)
        success=all(results.values())
    else:
        print("🔧 Running full pipeline (all steps + pip check + SageMaker tests)...")
        results=automation.run_automation_with_testing(enable_pip_check=True, enable_sagemaker_tests=True)
        success=all(results.values())
    exit(0 if success else 1)

if __name__ == "__main__":
    main()