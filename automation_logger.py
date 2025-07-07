import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import inspect
class AutomationLogger:
    """Centralized logging system for AutoGluon automation steps"""
    
    def __init__(self, current_version: str, step_name: str = None, custom_name: str = None):
        self.current_version = current_version
        if step_name is None and custom_name is None:
            step_name = self._auto_detect_step_name()
        elif custom_name:
            step_name = custom_name
        self.step_name = step_name
        self.logger = logging.getLogger(f"{step_name}_logger")
        self.setup_subprocess_logging()
        
    def _auto_detect_step_name(self) -> str:
        """Auto-detect the step name from the calling module"""
        try:
            frame = inspect.currentframe().f_back.f_back
            filename = frame.f_code.co_filename
            module_name = Path(filename).stem
            if 'self' in frame.f_locals:
                class_name = frame.f_locals['self'].__class__.__name__
                step_name = self._camel_to_snake(class_name)
            else:
                step_name = module_name
            self.logger.info(f"Auto-detected step name: {step_name}")
            return step_name
        except Exception as e:
            fallback_name = "automation_step"
            self.logger.warning(f"Could not auto-detect step name: {e}, using fallback: {fallback_name}")
            return fallback_name
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def setup_subprocess_logging(self):
        """Setup logging for subprocess commands with organized folder structure"""
        # Create main logs directory using absolute path
        current_working_dir = Path.cwd()
        base_logs_dir = current_working_dir / "logs"
        base_logs_dir.mkdir(exist_ok=True)
        step_logs_dir = base_logs_dir / self.step_name
        step_logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_filename = f"{self.step_name}_{timestamp}.txt"
        self.log_file_path = step_logs_dir / log_filename
        self.log_file_path.touch()
        self.logger.info(f"📝 Subprocess logs will be written to: {self.log_file_path}")
        with open(self.log_file_path, 'w') as f:
            f.write(f"{self.step_name.replace('_', ' ').title()} Automation Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Version: {self.current_version}\n")
            f.write(f"Step: {self.step_name}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_subprocess_output(self, command: str, result, step_description: str = ""):
        """Log subprocess command and its output to the log file"""
        with open(self.log_file_path, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {step_description}\n")
            f.write(f"Command: {command}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write(f"Working Directory: {os.getcwd()}\n")
            f.write("-" * 40 + "\n")
            if hasattr(result, 'stdout') and result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n")
            else:
                f.write("STDOUT: (no output)\n")
            if hasattr(result, 'stderr') and result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
                f.write("\n")
            else:
                f.write("STDERR: (no output)\n")
            
            f.write("=" * 80 + "\n\n")
    
    def run_subprocess_with_logging(self, command: List[str], step_description: str = "", capture_output: bool = True, **kwargs):
        """Run subprocess command with logging"""
        if 'text' not in kwargs:
            kwargs['text'] = True
        command_str = ' '.join(command)
        self.logger.info(f"🔧 Running: {command_str}")
        with open(self.log_file_path, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING: {step_description}\n")
            f.write(f"Command: {command_str}\n")
            f.write(f"Working Directory: {os.getcwd()}\n")
            f.write("-" * 40 + "\n")
        try:
            if capture_output:
                result = subprocess.run(command, capture_output=True, **kwargs)
                self.log_subprocess_output(command_str, result, step_description)
                return result
            else:
                with open(self.log_file_path, 'a') as log_file:
                    log_file.write("STDOUT/STDERR (real-time):\n")
                    log_file.flush()
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                        **kwargs
                    )
                    output_lines = []
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        print(line, end='')
                        log_file.write(line)
                        log_file.flush()
                        output_lines.append(line)
                    return_code = process.wait()
                    log_file.write(f"\n[Command completed with return code: {return_code}]\n")
                    log_file.write("=" * 80 + "\n\n")
                    log_file.flush()
                    class ProcessResult:
                        def __init__(self, returncode, stdout, stderr=""):
                            self.returncode = returncode
                            self.stdout = stdout
                            self.stderr = stderr
                    
                    return ProcessResult(return_code, ''.join(output_lines), "")            
        except Exception as e:
            with open(self.log_file_path, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {step_description} - EXCEPTION\n")
                f.write(f"Command: {command_str}\n")
                f.write(f"Exception: {str(e)}\n")
                f.write(f"Working Directory: {os.getcwd()}\n")
                f.write("=" * 80 + "\n\n")
            raise
    
    def log_step_start(self, step_description: str):
        """Log the start of a major step"""
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STEP START: {step_description}\n")
            f.write(f"{'='*60}\n\n")
    
    def log_step_complete(self, step_description: str, success: bool = True):
        """Log the completion of a major step"""
        status = "SUCCESS" if success else "FAILED"
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STEP {status}: {step_description}\n")
            f.write(f"{'='*60}\n\n")
    
    def log_automation_complete(self, success: bool = True):
        """Log the final completion of automation"""
        status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AUTOMATION {status}\n")
            f.write(f"Step: {self.step_name}\n")
            f.write(f"{'='*80}\n")
    
    def get_log_file_path(self) -> Path:
        """Get the current log file path"""
        return self.log_file_path
    
    def add_custom_log(self, message: str, level: str = "INFO"):
        """Add a custom log message"""
        with open(self.log_file_path, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level}: {message}\n")


class LoggerMixin:
    """Mixin class to easily add logging functionality to existing automation classes"""
    
    def setup_logging(self, current_version: str, step_name: str = None, custom_name: str = None):
        """Setup logging for the automation class"""
        self.automation_logger = AutomationLogger(current_version, step_name, custom_name)
        return self.automation_logger
    
    def run_subprocess_with_logging(self, command: List[str], step_description: str = "", capture_output: bool = True, **kwargs):
        """Delegate to automation logger"""
        if not hasattr(self, 'automation_logger'):
            raise RuntimeError("Logging not setup. Call setup_logging() first.")
        return self.automation_logger.run_subprocess_with_logging(command, step_description, capture_output, **kwargs)
    
    def log_step_start(self, step_description: str):
        """Delegate to automation logger"""
        if hasattr(self, 'automation_logger'):
            self.automation_logger.log_step_start(step_description)
    
    def log_step_complete(self, step_description: str, success: bool = True):
        """Delegate to automation logger"""
        if hasattr(self, 'automation_logger'):
            self.automation_logger.log_step_complete(step_description, success)
    
    def log_automation_complete(self, success: bool = True):
        """Delegate to automation logger"""
        if hasattr(self, 'automation_logger'):
            self.automation_logger.log_automation_complete(success)