import argparse
import logging
from steps_1_2_5 import Steps125Automation
from steps_3_4 import Steps34Automation  
from step_6 import Step6Automation

class AutoGluonReleaseAutomation:
    """Main orchestrator for AutoGluon release automation"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        self.current_version = current_version
        self.previous_version = previous_version
        self.fork_url = fork_url
        current_parts = current_version.split('.')
        previous_parts = previous_version.split('.')
        self.is_major_release = (current_parts[0] != previous_parts[0] or 
                               (current_parts[1] != previous_parts[1] and current_parts[2] == '0'))
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AutoGluon Release Automation - Version {current_version}")
        self.logger.info(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
        self.steps_125 = Steps125Automation(current_version, previous_version, fork_url)
        self.steps_34 = Steps34Automation(current_version, previous_version, fork_url)
        self.step_6 = Step6Automation(current_version, previous_version, fork_url)
        self.results = {}
        self.selected_images = None

    def run_automation(self, steps_only=None):
        """Run the complete automation or specific steps"""
        try:
            self.logger.info("Starting AutoGluon release automation...")
            if self._should_run_steps([1, 2, 5], steps_only):
                self.logger.info("Running Steps 1, 2, and 5...")
                step_125_results = self.steps_125.run_steps(steps_only)
                self.results.update(step_125_results)
            if self._should_run_steps([3, 4], steps_only):
                self.logger.info("Running Steps 3 and 4...")
                step_34_results = self.steps_34.run_steps(steps_only)
                self.results.update(step_34_results)
                self.selected_images = self.steps_34.get_selected_images()
            if self._should_run_steps([6], steps_only):
                self.logger.info("Running Step 6...")
                step_6_results = self.step_6.run_steps(steps_only)
                self.results.update(step_6_results)
            self._print_summary(steps_only)
            return self.results
        except Exception as e:
            self.logger.error(f"❌ Automation failed: {e}")
            self.logger.exception("Full error details:")
            raise

    def _should_run_steps(self, step_numbers, steps_only):
        """Check if any of the given step numbers should be run"""
        if not steps_only:
            return True
        return any(step in steps_only for step in step_numbers)

    def _print_summary(self, steps_only):
        """Print automation summary"""
        print("\n" + "="*60)
        
        if steps_only:
            completed_steps = [str(step) for step in sorted(steps_only) if self.results.get(step, False)]
            failed_steps = [str(step) for step in sorted(steps_only) if step in self.results and not self.results[step]]
            
            if completed_steps:
                print(f"✅ AutoGluon {self.current_version} automation completed")
                print(f"   Completed steps: {', '.join(completed_steps)}")
            if failed_steps:
                print(f"❌ Failed steps: {', '.join(failed_steps)}")
        else:
            successful_steps = [str(k) for k, v in self.results.items() if v]
            failed_steps = [str(k) for k, v in self.results.items() if not v]
            
            if len(successful_steps) == 6:
                print(f"✅ AutoGluon {self.current_version} release automation completed successfully")
            else:
                print(f"⚠️  AutoGluon {self.current_version} automation completed with issues")
                if successful_steps:
                    print(f"   Successful steps: {', '.join(successful_steps)}")
                if failed_steps:
                    print(f"   Failed steps: {', '.join(failed_steps)}")
        
        print(f"Release type: {'Major' if self.is_major_release else 'Minor'}")
        print(f"Branch: autogluon-{self.current_version}-release")
        
        if self.selected_images:
            print(f"Selected PyTorch: {self.selected_images.get('pytorch_version', 'N/A')}")
            print(f"Selected CUDA: {self.selected_images.get('cuda_version', 'N/A')}")
        
        print("="*60)

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
    parser = argparse.ArgumentParser(description='AutoGluon DLC Release Automation')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.2)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.1)') 
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    parser.add_argument('--steps-only', nargs='+', type=int, help='Run only specific steps (e.g., 1 2 3 4 5 6)')
    parser.add_argument('--steps-125', action='store_true', help='Run only steps 1, 2, and 5')
    parser.add_argument('--steps-34', action='store_true', help='Run only steps 3 and 4')
    parser.add_argument('--step-6', action='store_true', help='Run only step 6')
    args = parser.parse_args()
    
    automation = AutoGluonReleaseAutomation(
        args.current_version,
        args.previous_version, 
        args.fork_url
    )
    if args.steps_125:
        automation.run_steps_1_2_5()
    elif args.steps_34:
        automation.run_steps_3_4()
    elif args.step_6:
        automation.run_step_6()
    else:
        automation.run_automation(steps_only=args.steps_only)

if __name__ == "__main__":
    main()