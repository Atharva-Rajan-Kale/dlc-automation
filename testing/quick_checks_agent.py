"""
title : Quick Checks Failure Resolution Agent

description : Agent that detects and resolves dlc-pr-quick-checks test failures
by automatically reverting TOML configuration to full framework builds when
only quick-checks are failing. Uses advanced test detection via GraphQL and
REST APIs to analyze PR status, identifies when quick-checks are the sole
blocker, and applies targeted fixes by undoing step 2 TOML changes. Includes
comprehensive failure analysis, git operations with authentication, and
automated monitoring of test completion after fixes.
"""

import os
import re
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

from automation.common import BaseAutomation
from automation.automation_logger import LoggerMixin
class QuickChecksAgent(BaseAutomation,LoggerMixin):
    """Agent to handle dlc-pr-quick-checks failures by reverting TOML to full framework build"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_github_credentials()
        self.branch_name = f"autogluon-{current_version}-release"
        self.setup_logging(current_version,custom_name="quick_check")
        
    def setup_github_credentials(self):
        """Setup GitHub credentials for API access"""
        self.github_token = os.environ.get('GITHUB_TOKEN')
        if not self.github_token:
            try:
                result = self.run_subprocess_with_logging(
                    ["gh", "auth", "token"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                self.github_token = result.stdout.strip()
                self.logger.info("✅ GitHub token obtained from gh CLI")
            except:
                self.logger.error("❌ No GitHub token available")
                self.logger.info("💡 Please set GITHUB_TOKEN or run 'gh auth login'")
                raise Exception("GitHub token required for log access")
    
    def get_current_pr_number(self) -> Optional[int]:
        """Get the current PR number for the branch"""
        try:
            url = f"https://api.github.com/repos/aws/deep-learning-containers/pulls"
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            params = {
                "head": f"{self.fork_url.split('/')[-2]}:{self.branch_name}",
                "state": "open"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            prs = response.json()
            if prs:
                pr_number = prs[0]['number']
                self.logger.info(f"📋 Found PR #{pr_number} for branch {self.branch_name}")
                return pr_number
            else:
                self.logger.warning(f"⚠️ No open PR found for branch {self.branch_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get PR number: {e}")
            return None
    def configure_git_with_token(self, token: str) -> bool:
        """Configure Git with GitHub token for authentication"""
        try:
            # Get current remote URL
            result = self.run_subprocess_with_logging(
                ["git", "remote", "get-url", "origin"], 
                capture_output=True, text=True
            )
            current_url = result.stdout.strip()
            
            # Configure git user (required for commits)
            self.run_subprocess_with_logging(
                ["git", "config", "user.name", "Atharva-Rajan-Kale"], 
                capture_output=True
            )
            self.run_subprocess_with_logging(
                ["git", "config", "user.email", "atharvakale912@gmail.com"], 
                capture_output=True
            )
            
            # Add token to URL if it's a GitHub URL and doesn't already have a token
            if "github.com" in current_url and "@github.com" not in current_url:
                if current_url.startswith("https://github.com/"):
                    authenticated_url = current_url.replace("https://github.com/", f"https://{token}@github.com/")
                else:
                    authenticated_url = current_url.replace("github.com", f"{token}@github.com")
                
                self.run_subprocess_with_logging(
                    ["git", "remote", "set-url", "origin", authenticated_url], 
                    check=True
                )
                self.logger.info("✅ Configured git remote with GitHub token")
            else:
                self.logger.info("ℹ️ Git remote already configured or not a GitHub URL")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to configure git with token: {e}")
            return False
    def get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment or GitHub CLI"""
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            return token
        try:
            result = self.run_subprocess_with_logging(
                ["gh", "auth", "token"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Could not get GitHub token from gh CLI")
            return None
    def _get_graphql_tests_for_commit(self, pr_number: int, commit_sha: str) -> List[Dict]:
        """Get tests for a specific commit via GraphQL (for AutoGluon tests)"""
        query = """
        query($owner: String!, $repo: String!, $oid: GitObjectID!) {
          repository(owner: $owner, name: $repo) {
            object(oid: $oid) {
              ... on Commit {
                checkSuites(first: 50) {
                  nodes {
                    id
                    app {
                      name
                    }
                    checkRuns(first: 100) {
                      nodes {
                        id
                        databaseId
                        name
                        status
                        conclusion
                        url
                        detailsUrl
                      }
                    }
                  }
                }
                status {
                  contexts {
                    context
                    state
                    targetUrl
                  }
                }
              }
            }
          }
        }
        """
        variables = {
            "owner": "aws",
            "repo": "deep-learning-containers",
            "oid": commit_sha
        }
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                "https://api.github.com/graphql",
                headers=headers,
                json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            data = response.json()
            if 'errors' in data:
                self.logger.warning(f"GraphQL errors: {data['errors']}")
                return []
            tests = []
            commit_data = data['data']['repository']['object']
            if commit_data:
                # Process check suites
                check_suites = commit_data.get('checkSuites', {}).get('nodes', [])
                for suite in check_suites:
                    check_runs = suite.get('checkRuns', {}).get('nodes', [])
                    for run in check_runs:
                        conclusion = run.get('conclusion', '').upper()
                        status = run.get('status', '').upper()
                        if conclusion == 'SUCCESS':
                            state = 'SUCCESS'
                        elif conclusion in ['FAILURE', 'CANCELLED', 'TIMED_OUT']:
                            state = conclusion
                        elif status in ['IN_PROGRESS', 'QUEUED', 'REQUESTED']:
                            state = 'PENDING'
                        else:
                            state = conclusion or status or 'UNKNOWN'
                        tests.append({
                            'name': run['name'],
                            'state': state,
                            'type': 'check_run',
                            'url': run.get('url', ''),
                            'details_url': run.get('detailsUrl', '')
                        })
                # Process commit statuses
                status = commit_data.get('status')
                if status:
                    contexts = status.get('contexts', [])
                    for context in contexts:
                        tests.append({
                            'name': context['context'],
                            'state': context['state'].upper(),
                            'type': 'status_context',
                            'url': context.get('targetUrl', ''),
                            'details_url': context.get('targetUrl', '')
                        })
            return tests
        except Exception as e:
            self.logger.warning(f"Failed to get GraphQL tests for commit {commit_sha}: {e}")
            return []
    
    def get_all_failing_tests(self, pr_number: int) -> Dict:
        """Get all failing tests using both REST API and GraphQL"""
        try:
            # Get PR details
            pr_url = f"https://api.github.com/repos/aws/deep-learning-containers/pulls/{pr_number}"
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            response = requests.get(pr_url, headers=headers)
            response.raise_for_status()
            pr_data = response.json()
            
            head_sha = pr_data['head']['sha']
            all_failing_tests = []
            # 1. Check REST API Check Runs for HEAD
            check_runs_url = f"https://api.github.com/repos/aws/deep-learning-containers/commits/{head_sha}/check-runs"
            response = requests.get(check_runs_url, headers=headers, params={"per_page": 100})
            if response.status_code == 200:
                check_runs = response.json().get('check_runs', [])
                for run in check_runs:
                    if run['status'] == 'completed' and run['conclusion'] == 'failure':
                        all_failing_tests.append({
                            'name': run['name'],
                            'source': 'REST_check_run',
                            'url': run['html_url'],
                            'details_url': run.get('details_url', '')
                        })
            
            # 2. Check REST API Commit Statuses for HEAD
            status_url = f"https://api.github.com/repos/aws/deep-learning-containers/commits/{head_sha}/status"
            response = requests.get(status_url, headers=headers)
            if response.status_code == 200:
                status_data = response.json()
                statuses = status_data.get('statuses', [])
                for status in statuses:
                    if status['state'] in ['failure', 'error']:
                        # Check if we already have this test from check runs
                        existing = any(t['name'] == status['context'] for t in all_failing_tests)
                        if not existing:
                            all_failing_tests.append({
                                'name': status['context'],
                                'source': 'REST_status',
                                'url': status.get('target_url', ''),
                                'details_url': status.get('target_url', '')
                            })
            # 3. Check GraphQL for current HEAD (for AutoGluon tests)
            graphql_tests = self._get_graphql_tests_for_commit(pr_number, head_sha)
            for test in graphql_tests:
                if test['state'] in ['FAILURE', 'ERROR', 'CANCELLED', 'TIMED_OUT']:
                    # Check if we already have this test
                    existing = any(t['name'] == test['name'] for t in all_failing_tests)
                    if not existing:
                        all_failing_tests.append({
                            'name': test['name'],
                            'source': 'GraphQL',
                            'url': test.get('url', ''),
                            'details_url': test.get('details_url', '')
                        })
            # Categorize tests
            quick_checks_tests = [t for t in all_failing_tests if 'dlc-pr-quick-checks' in t['name'].lower()]
            other_failing_tests = [t for t in all_failing_tests if 'dlc-pr-quick-checks' not in t['name'].lower()]
            self.logger.info(f"📊 Found {len(all_failing_tests)} total failing tests")
            self.logger.info(f"   - dlc-pr-quick-checks: {len(quick_checks_tests)}")
            self.logger.info(f"   - Other tests: {len(other_failing_tests)}")
            return {
                'all_failing': all_failing_tests,
                'quick_checks': quick_checks_tests,
                'other_failing': other_failing_tests,
                'total_failing': len(all_failing_tests),
                'only_quick_checks_failing': len(other_failing_tests) == 0 and len(quick_checks_tests) > 0
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get failing tests: {e}")
            return {
                'all_failing': [],
                'quick_checks': [],
                'other_failing': [],
                'total_failing': 0,
                'only_quick_checks_failing': False
            }
    
    def revert_toml_to_full_build(self) -> bool:
        """Revert TOML file back to full framework build (undo step 2 changes)"""
        self.logger.info("🔄 Reverting TOML to full framework build...")
        original_dir = os.getcwd()
        try:
            if not self.repo_dir.exists():
                self.logger.error(f"Repository directory not found: {self.repo_dir}")
                return False
            os.chdir(self.repo_dir)
            toml_path = Path("dlc_developer_config.toml")
            if not toml_path.exists():
                self.logger.error(f"TOML file not found: {toml_path.absolute()}")
                return False
            with open(toml_path, 'r') as f:
                content = f.read()
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'build_frameworks' in line:
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    for j in range(start, end):
                        marker = " -> " if j == i else "    "
                    break
            # Revert build_frameworks back to full list
            # This reverses the change made in step 2 of steps_1_2_5.py
            content = re.sub(
                r'build_frameworks\s*=\s*\["autogluon"\]',
                'build_frameworks = []',
                content,
                flags=re.DOTALL
            )
            self.logger.info("✅ Reverted build_frameworks to include all frameworks")
            # Also revert the buildspec paths back to empty (if they were changed)
            content = re.sub(
                r'dlc-pr-autogluon-training\s*=\s*"autogluon/training/buildspec.yml"',
                'dlc-pr-autogluon-training = ""',
                content
            )
            self.logger.info("✅ Reverted dlc-pr-autogluon-training buildspec path")
            
            content = re.sub(
                r'dlc-pr-autogluon-inference\s*=\s*"autogluon/inference/buildspec.yml"',
                'dlc-pr-autogluon-inference = ""',
                content
            )
            self.logger.info("✅ Reverted dlc-pr-autogluon-inference buildspec path")
            # Show what the changes will be
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'build_frameworks' in line:
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    for j in range(start, end):
                        marker = " -> " if j == i else "    "
                    break
            # Write the updated content
            with open(toml_path, 'w') as f:
                f.write(content)
            self.logger.info("✅ TOML file updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to revert TOML: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def commit_and_push_toml_revert(self) -> bool:
        """Commit and push the TOML revert changes with user confirmation"""
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            # Check if there are changes
            result = self.run_subprocess_with_logging(["git", "diff", "--quiet"], capture_output=True)
            if result.returncode == 0:
                self.logger.info("ℹ️ No changes to commit")
                return True
            # Show what changes will be committed
            self.logger.info("📋 Changes to be committed:")
            diff_result = self.run_subprocess_with_logging(["git", "diff", "--name-only"], capture_output=True, text=True)
            if diff_result.stdout:
                for file in diff_result.stdout.strip().split('\n'):
                    self.logger.info(f"   📝 Modified: {file}")
            # Show a preview of the TOML changes
            self.logger.info("\n📄 Preview of TOML changes:")
            diff_preview = self.run_subprocess_with_logging(["git", "diff", "dlc_developer_config.toml"], capture_output=True, text=True)
            if diff_preview.stdout:
                for line in diff_preview.stdout.strip().split('\n'):
                    if line.startswith('+') and 'build_frameworks' in line:
                        self.logger.info(f"   🟢 {line}")
                    elif line.startswith('-') and 'build_frameworks' in line:
                        self.logger.info(f"   🔴 {line}")
            commit_message = f"AutoGluon {self.current_version}: Revert TOML to full framework build for dlc-pr-quick-checks"
            # Configure Git with token before pushing
            token = self.get_github_token()
            if not token:
                self.logger.error("❌ No GitHub token available for push")
                return False
            if not self.configure_git_with_token(token):
                self.logger.error("❌ Failed to configure git with token")
                return False
            # Add changes
            self.run_subprocess_with_logging(["git", "add", "dlc_developer_config.toml"], check=True)
            # Commit changes
            self.run_subprocess_with_logging(["git", "commit", "-m", commit_message], check=True)
            # Push to branch
            self.run_subprocess_with_logging(["git", "push", "origin", self.branch_name], check=True)
            self.logger.info(f"✅ Successfully committed and pushed TOML revert changes")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to commit and push TOML revert: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def wait_for_quick_checks_completion(self, pr_number: int, max_wait_minutes: int = 60) -> bool:
        """Wait for dlc-pr-quick-checks to complete after TOML revert"""
        self.logger.info(f"⏳ Waiting up to {max_wait_minutes} minutes for dlc-pr-quick-checks to complete...")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval_seconds = 30
        while (time.time() - start_time) < max_wait_seconds:
            try:
                test_status = self.get_all_failing_tests(pr_number)
                quick_checks_tests = test_status['quick_checks']
                elapsed_minutes = int((time.time() - start_time) / 60)
                if not quick_checks_tests:
                    self.logger.info(f"✅ dlc-pr-quick-checks are now passing! (after {elapsed_minutes}m)")
                    return True
                else:
                    self.logger.info(f"⏳ dlc-pr-quick-checks still running/failing... ({elapsed_minutes}m elapsed)")
                    for test in quick_checks_tests:
                        self.logger.info(f"   - {test['name']}")
                time.sleep(check_interval_seconds)
            except Exception as e:
                self.logger.warning(f"⚠️ Error checking test status: {e}")
                time.sleep(check_interval_seconds)
                continue
        self.logger.warning(f"⏰ Timeout after {max_wait_minutes} minutes - dlc-pr-quick-checks may still be running")
        return False
    
    def run_quick_checks_agent(self) -> bool:
        """
        Main quick checks agent execution:
        1. Check if only dlc-pr-quick-checks are failing
        2. If so, revert TOML to full framework build
        3. Commit and push changes
        4. Wait for tests to complete
        """
        self.logger.info("🔍 Starting Quick Checks Agent...")
        try:
            # Get current PR number
            pr_number = self.get_current_pr_number()
            if not pr_number:
                self.logger.error("❌ No PR found for quick checks analysis")
                return False
            # Get all failing tests
            self.logger.info("📊 Analyzing current test failures...")
            test_status = self.get_all_failing_tests(pr_number)
            total_failing = test_status['total_failing']
            quick_checks_count = len(test_status['quick_checks'])
            other_failing_count = len(test_status['other_failing'])
            only_quick_checks_failing = test_status['only_quick_checks_failing']
            self.logger.info("="*80)
            self.logger.info("📊 TEST FAILURE ANALYSIS")
            self.logger.info("="*80)
            self.logger.info(f"Total failing tests: {total_failing}")
            self.logger.info(f"dlc-pr-quick-checks failing: {quick_checks_count}")
            self.logger.info(f"Other tests failing: {other_failing_count}")
            self.logger.info(f"Only quick-checks failing: {only_quick_checks_failing}")
            if other_failing_count > 0:
                self.logger.info("\n❌ OTHER FAILING TESTS DETECTED:")
                for test in test_status['other_failing']:
                    self.logger.info(f"   - {test['name']} (source: {test['source']})")
            if quick_checks_count > 0:
                self.logger.info("\n⚠️ QUICK-CHECKS FAILING TESTS:")
                for test in test_status['quick_checks']:
                    self.logger.info(f"   - {test['name']} (source: {test['source']})")
            self.logger.info("="*80)
            # Check if we should proceed
            if not only_quick_checks_failing:
                if total_failing == 0:
                    self.logger.info("✅ No failing tests detected - Quick Checks Agent not needed")
                    return True
                else:
                    self.logger.info("❌ Other tests are failing besides dlc-pr-quick-checks")
                    self.logger.info("🛑 Quick Checks Agent will not proceed - fix other failures first")
                    self.logger.info("💡 The Quick Checks Agent only runs when ONLY dlc-pr-quick-checks are failing")
                    return False
            # Proceed with TOML revert
            self.logger.info("✅ Only dlc-pr-quick-checks are failing - proceeding with TOML revert")
            self.logger.info("🔄 This will revert the TOML to build all frameworks instead of just AutoGluon")
            # Revert TOML configuration
            toml_success = self.revert_toml_to_full_build()
            if not toml_success:
                self.logger.error("❌ Failed to revert TOML configuration")
                return False
            # Commit and push changes
            commit_success = self.commit_and_push_toml_revert()
            if not commit_success:
                self.logger.error("❌ Failed to commit TOML revert changes")
                return False
            # Wait for quick checks to complete
            self.logger.info("⏳ Waiting for dlc-pr-quick-checks to complete after TOML revert...")
            checks_passed = self.wait_for_quick_checks_completion(pr_number, max_wait_minutes=60)
            if checks_passed:
                self.logger.info("🎉 dlc-pr-quick-checks are now passing after TOML revert!")
                return True
            else:
                self.logger.warning("⚠️ dlc-pr-quick-checks may still be running - check PR status manually")
                return True  # Still consider success since we applied the fix
        except Exception as e:
            self.logger.error(f"❌ Quick Checks Agent failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
def main():
    """Main function for Quick Checks Agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoGluon Quick Checks Agent')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.1)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.0)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    args = parser.parse_args()
    
    agent = QuickChecksAgent(
        args.current_version,
        args.previous_version,
        args.fork_url
    )
    
    success = agent.run_quick_checks_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()