import os
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Optional, Dict, List
import logging
from automation_logger import LoggerMixin

class GitHubPRAutomation(LoggerMixin):
    """Handles automatic PR creation to upstream repository"""
    
    def __init__(self, current_version: str, fork_url: str, repo_dir: Path):
        self.current_version = current_version
        self.fork_url = fork_url
        self.repo_dir = repo_dir
        self.branch_name = f"autogluon-{current_version}-release"
        self.logger = logging.getLogger(__name__)
        if "github.com/" in fork_url:
            self.fork_owner = fork_url.split("github.com/")[1].split("/")[0]
        else:
            raise ValueError("Invalid GitHub fork URL format")
        self.upstream_owner = "aws"
        self.repo_name = "deep-learning-containers"
        self.setup_logging(current_version,custom_name="github_pr")
        
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
    
    def format_code_with_black(self) -> bool:
        """Format code with black before creating PR"""
        self.logger.info("🎨 Formatting code with black...")
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            result = self.run_subprocess_with_logging(
                ["black", "-l", "100", "."],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info("✅ Code formatting completed successfully")
                result = self.run_subprocess_with_logging(
                    ["git", "diff", "--quiet"],
                    capture_output=True
                )
                if result.returncode != 0:  
                    self.logger.info("📝 Committing formatting changes...")
                    self.run_subprocess_with_logging(["git", "add", "."], check=True)
                    self.run_subprocess_with_logging([
                        "git", "commit", "-m", 
                        f"Add Autogluon v{self.current_version}"
                    ], check=True)
                    self.logger.info("✅ Formatting changes committed")
                else:
                    self.logger.info("ℹ️ No formatting changes needed")
                return True
            else:
                self.logger.warning(f"⚠️ Black formatting had issues: {result.stderr}")
                return True
        except FileNotFoundError:
            self.logger.warning("⚠️ Black not found. Install with: pip install black")
            return True  
        except Exception as e:
            self.logger.warning(f"⚠️ Code formatting failed: {e}")
            return True  
        finally:
            os.chdir(original_dir)

    def push_branch_to_fork(self) -> bool:
        """Push the current branch to the fork"""
        self.logger.info(f"🚀 Pushing branch {self.branch_name} to fork...")
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            self.run_subprocess_with_logging(["git", "checkout", self.branch_name], check=True)
            if not self.format_code_with_black():
                self.logger.warning("⚠️ Code formatting failed, continuing with push...")
            result = self.run_subprocess_with_logging(
                ["git", "push", "origin", self.branch_name], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"✅ Successfully pushed {self.branch_name} to fork")
                return True
            else:
                if "already exists" in result.stderr:
                    self.logger.info(f"ℹ️ Branch {self.branch_name} already exists, force pushing...")
                    self.run_subprocess_with_logging(
                        ["git", "push", "--force", "origin", self.branch_name], 
                        check=True
                    )
                    return True
                else:
                    self.logger.error(f"❌ Failed to push branch: {result.stderr}")
                    return False
        except Exception as e:
            self.logger.error(f"❌ Failed to push branch: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def check_existing_pr(self, token: str, include_closed: bool = False) -> Optional[Dict]:
        """Check if PR already exists for this branch (open or closed)"""
        url = f"https://api.github.com/repos/{self.upstream_owner}/{self.repo_name}/pulls"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        states = ["open"]
        if include_closed:
            states.append("closed")
        for state in states:
            params = {
                "head": f"{self.fork_owner}:{self.branch_name}",
                "state": state
            }
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                prs = response.json()
                if prs:
                    pr = prs[0]  
                    pr['original_state'] = state  
                    return pr
            except Exception as e:
                self.logger.error(f"❌ Failed to check {state} PRs: {e}")
                continue
        return None

    def reopen_pr(self, pr_number: int, token: str) -> bool:
        """Reopen a closed PR"""
        url = f"https://api.github.com/repos/{self.upstream_owner}/{self.repo_name}/pulls/{pr_number}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        data = {"state": "open"}
        try:
            response = requests.patch(url, headers=headers, json=data)
            response.raise_for_status()
            self.logger.info(f"✅ Reopened PR #{pr_number}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to reopen PR #{pr_number}: {e}")
            return False
    
    def get_pr_template(self) -> str:
        """Fetch PR template from upstream repository or use local template as fallback"""
        try:
            template_url = f"https://raw.githubusercontent.com/{self.upstream_owner}/{self.repo_name}/master/.github/PULL_REQUEST_TEMPLATE.md"
            response = requests.get(template_url, timeout=10)
            if response.status_code == 200:
                self.logger.info("✅ Fetched PR template from upstream repository")
                return response.text
            else:
                self.logger.info(f"ℹ️ PR template not found at upstream, using local template")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to fetch upstream PR template: {e}")
        try:
            template_path = self.repo_dir / ".github" / "PULL_REQUEST_TEMPLATE.md"
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info("✅ Using local PR template from .github/PULL_REQUEST_TEMPLATE.md")
            return content
        except Exception as e:
            self.logger.error(f"❌ Failed to read local PR template at {template_path}: {e}")
            raise Exception("Could not find PR template - ensure .github/PULL_REQUEST_TEMPLATE.md exists in your repository")

    def _is_autogluon_related_test(self, test_name: str) -> bool:
        """Enhanced AutoGluon test detection based on testrunner.py patterns"""
        test_name_lower = test_name.lower()
        if 'autogluon' in test_name_lower:
            return True
        autogluon_patterns = [
            'dlc-pr-autogluon',  
            'autogluon-inference',
            'autogluon-training',
            'ag-inference',
            'ag-training'
        ]
        sagemaker_patterns = [
            'sagemaker-local-test',
            'sagemaker-test', 
            'sanity-test',
            'security-test'
        ]
        for pattern in autogluon_patterns:
            if pattern in test_name_lower:
                return True
        for pattern in sagemaker_patterns:
            if pattern in test_name_lower:
                if any(ag_pattern in test_name_lower for ag_pattern in ['autogluon', '_ag_', '-ag-']):
                    return True
        return False

    def _determine_test_status(self, conclusion: str, status: str) -> tuple:
        """Determine test status and type from conclusion and status"""
        if conclusion == 'success':
            return 'success', 'passing'
        elif conclusion in ['failure', 'cancelled', 'timed_out']:
            return conclusion, 'failing'
        elif status in ['in_progress', 'queued', 'requested', 'waiting', 'pending'] or conclusion is None:
            return status if status else 'queued', 'pending'
        elif conclusion in ['neutral', 'skipped']:
            return conclusion, 'passing'  
        else:
            return f"{status}/{conclusion}", 'unknown'

    def _determine_status_state(self, state: str) -> tuple:
        """Determine test status and type from commit status state"""
        if state == 'success':
            return 'success', 'passing'
        elif state in ['failure', 'error']:
            return state, 'failing'
        elif state == 'pending':
            return 'pending', 'pending'
        else:
            return state, 'unknown'

    def create_pr_via_api(self, token: str) -> bool:
        """Create PR using GitHub API or reopen existing closed PR"""
        existing_pr = self.check_existing_pr(token, include_closed=True)
        if existing_pr:
            pr_url = existing_pr.get('html_url')
            pr_number = existing_pr.get('number')
            original_state = existing_pr.get('original_state')
            if original_state == 'open':
                self.logger.info(f"✅ PR already exists and is open: #{pr_number}")
                self.logger.info(f"🔗 PR URL: {pr_url}")
                return True
            elif original_state == 'closed':
                self.logger.info(f"🔄 Found closed PR #{pr_number}, reopening...")
                if self.reopen_pr(pr_number, token):
                    self.logger.info(f"✅ Successfully reopened PR #{pr_number}")
                    self.logger.info(f"🔗 PR URL: {pr_url}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Failed to reopen PR #{pr_number}, will create new one")
        url = f"https://api.github.com/repos/{self.upstream_owner}/{self.repo_name}/pulls"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        title = f"AutoGluon {self.current_version} DLC Release"
        body = self.get_pr_template()
        data = {
            "title": title,
            "body": body,
            "head": f"{self.fork_owner}:{self.branch_name}",
            "base": "master"
            
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            pr_data = response.json()
            pr_url = pr_data.get('html_url')
            pr_number = pr_data.get('number')
            self.logger.info(f"✅ Successfully created PR #{pr_number}")
            self.logger.info(f"🔗 PR URL: {pr_url}")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                error_data = e.response.json()
                if "pull request already exists" in error_data.get('message', '').lower():
                    self.logger.info("✅ PR already exists")
                    return True
            self.logger.error(f"❌ Failed to create PR: {e.response.text}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to create PR: {e}")
            return False
    
    def get_pr_test_status(self, pr_number: int, token: str) -> Dict:
        """Get test status for a specific PR - now with GraphQL for complete coverage"""
        try:
            pr_url = f"https://api.github.com/repos/{self.upstream_owner}/{self.repo_name}/pulls/{pr_number}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            response = requests.get(pr_url, headers=headers)
            response.raise_for_status()
            pr_data = response.json()
            head_sha = pr_data['head']['sha']
            merge_commit_sha = pr_data.get('merge_commit_sha')
            self.logger.info(f"🔍 Getting comprehensive test status for PR #{pr_number}")
            self.logger.info(f"   Head SHA: {head_sha}")
            self.logger.info(f"   Merge SHA: {merge_commit_sha}")
            all_tests = {}  
            self.logger.info("🔍 Using GraphQL for comprehensive test detection...")
            graphql_tests = self._get_tests_via_graphql(pr_number, token)
            for test in graphql_tests:
                test_name = test['name']
                is_autogluon = self._is_autogluon_related_test(test_name)
                if is_autogluon:
                    self.logger.info(f"   🎯 FOUND AutoGluon test via GraphQL: {test_name} - {test['state']}")
                all_tests[test_name] = {
                    'name': test_name,
                    'status': test['state'].lower(),
                    'type': 'passing' if test['state'] == 'SUCCESS' else 'failing' if test['state'] in ['FAILURE', 'ERROR'] else 'pending',
                    'source': 'graphql',
                    'is_autogluon': is_autogluon
                }
            shas_to_check = [head_sha]
            if merge_commit_sha and merge_commit_sha != head_sha:
                shas_to_check.append(merge_commit_sha)
            for sha_index, sha in enumerate(shas_to_check):
                self.logger.info(f"🔍 Supplementing with REST API for SHA {sha_index + 1}/{len(shas_to_check)}: {sha}")
                try:
                    check_runs_url = f"https://api.github.com/repos/{self.upstream_owner}/{self.repo_name}/commits/{sha}/check-runs"
                    page = 1
                    while page <= 10:  
                        params = {"page": page, "per_page": 100}
                        response = requests.get(check_runs_url, headers=headers, params=params)
                        response.raise_for_status()
                        check_runs_data = response.json()
                        if not check_runs_data.get('check_runs'):
                            break
                        for check in check_runs_data['check_runs']:
                            check_name = check['name']
                            if check_name not in all_tests:  
                                is_autogluon = self._is_autogluon_related_test(check_name)
                                test_status, test_type = self._determine_test_status(check['conclusion'], check['status'])
                                all_tests[check_name] = {
                                    'name': check_name,
                                    'status': test_status,
                                    'type': test_type,
                                    'source': f'rest_check_run_{sha[:8]}',
                                    'is_autogluon': is_autogluon
                                }
                        page += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting check runs for {sha}: {e}")
                try:
                    status_url = f"https://api.github.com/repos/{self.upstream_owner}/{self.repo_name}/commits/{sha}/status"
                    response = requests.get(status_url, headers=headers)
                    response.raise_for_status()
                    status_data = response.json()
                    for status_check in status_data.get('statuses', []):
                        check_name = status_check['context']
                        if check_name not in all_tests:  
                            is_autogluon = self._is_autogluon_related_test(check_name)
                            test_status, test_type = self._determine_status_state(status_check['state'])
                            all_tests[check_name] = {
                                'name': check_name,
                                'status': test_status,
                                'type': test_type,
                                'source': f'rest_status_{sha[:8]}',
                                'is_autogluon': is_autogluon
                            }
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting statuses for {sha}: {e}")
            passing_tests = [test for test in all_tests.values() if test['type'] == 'passing']
            failing_tests = [test for test in all_tests.values() if test['type'] == 'failing']
            pending_tests = [test for test in all_tests.values() if test['type'] == 'pending']
            autogluon_tests = {name: test for name, test in all_tests.items() if test.get('is_autogluon', False)}
            autogluon_passing = [test for test in autogluon_tests.values() if test['type'] == 'passing']
            autogluon_failing = [test for test in autogluon_tests.values() if test['type'] == 'failing']
            autogluon_pending = [test for test in autogluon_tests.values() if test['type'] == 'pending']
            self.logger.info(f"📊 COMPREHENSIVE TEST ANALYSIS:")
            self.logger.info(f"   Total unique tests found: {len(all_tests)}")
            self.logger.info(f"   Passing: {len(passing_tests)}")
            self.logger.info(f"   Failing: {len(failing_tests)}")
            self.logger.info(f"   Pending: {len(pending_tests)}")
            self.logger.info(f"🎯 AUTOGLUON TEST ANALYSIS:")
            self.logger.info(f"   AutoGluon tests found: {len(autogluon_tests)}")
            self.logger.info(f"   AutoGluon passing: {len(autogluon_passing)}")
            self.logger.info(f"   AutoGluon failing: {len(autogluon_failing)}")
            self.logger.info(f"   AutoGluon pending: {len(autogluon_pending)}")
            if autogluon_tests:
                self.logger.info(f"📋 All AutoGluon tests found:")
                for name, test in sorted(autogluon_tests.items()):
                    self.logger.info(f"   - {test['name']} ({test['type']}: {test['status']}) [{test['source']}]")
            else:
                self.logger.warning("❌ NO AUTOGLUON TESTS FOUND!")
            if failing_tests:
                self.logger.info(f"❌ All failing tests:")
                for test in failing_tests:
                    self.logger.info(f"   - {test['name']} ({test['status']}) [{test['source']}]")
            return {
                'pr_number': pr_number,
                'head_sha': head_sha,
                'merge_commit_sha': merge_commit_sha,
                'passing': passing_tests,
                'failing': failing_tests,
                'pending': pending_tests,
                'autogluon_tests': list(autogluon_tests.values()),
                'autogluon_passing': autogluon_passing,
                'autogluon_failing': autogluon_failing,
                'autogluon_pending': autogluon_pending,
                'total_passing': len(passing_tests),
                'total_failing': len(failing_tests),
                'total_pending': len(pending_tests),
                'total_autogluon': len(autogluon_tests)
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get PR test status: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'pr_number': pr_number,
                'error': str(e),
                'passing': [],
                'failing': [],
                'pending': [],
                'autogluon_tests': []
            }

    def _get_tests_via_graphql(self, pr_number: int, token: str) -> List[Dict]:
        """Get tests using GraphQL API - this finds the AutoGluon tests that REST API misses"""
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              headRefOid
              mergeCommit {
                oid
              }
              commits(last: 10) {
                nodes {
                  commit {
                    oid
                    checkSuites(first: 50) {
                      nodes {
                        app {
                          name
                        }
                        checkRuns(first: 100) {
                          nodes {
                            name
                            status
                            conclusion
                          }
                        }
                      }
                    }
                    status {
                      contexts {
                        context
                        state
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        variables = {
            "owner": self.upstream_owner,
            "repo": self.repo_name,
            "number": pr_number
        }
        headers = {
            "Authorization": f"Bearer {token}",
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
                self.logger.error(f"❌ GraphQL errors: {data['errors']}")
                return []
            tests = []
            pr_data = data['data']['repository']['pullRequest']
            commits = pr_data['commits']['nodes']
            self.logger.info(f"📊 GraphQL found {len(commits)} commits to analyze")
            for commit_node in commits:
                commit = commit_node['commit']
                oid = commit['oid']
                check_suites = commit.get('checkSuites', {}).get('nodes', [])
                for suite in check_suites:
                    app_name = suite.get('app', {}).get('name', 'Unknown')
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
                            'commit_oid': oid,
                            'app': app_name,
                            'type': 'check_run'
                        })
                status = commit.get('status')
                if status:
                    contexts = status.get('contexts', [])
                    for context in contexts:
                        tests.append({
                            'name': context['context'],
                            'state': context['state'].upper(),
                            'commit_oid': oid,
                            'type': 'status_context'
                        })
            self.logger.info(f"📊 GraphQL extracted {len(tests)} total tests")
            return tests
        except Exception as e:
            self.logger.error(f"❌ GraphQL query failed: {e}")
            return []

    def monitor_pr_status(self, pr_number: int = None, show_pending: bool = True) -> Dict:
        """Monitor the status of the current PR or a specific PR number"""
        token = self.get_github_token()
        if not token:
            self.logger.error("❌ No GitHub token available for monitoring")
            return {}
        if pr_number is None:
            try:
                existing_pr = self.check_existing_pr(token, include_closed=False)
                if existing_pr:
                    pr_number = existing_pr['number']
                else:
                    self.logger.error("❌ No PR found for current branch")
                    return {}
            except Exception as e:
                self.logger.error(f"❌ Failed to find PR: {e}")
                return {}
        self.logger.info(f"📊 Checking test status for PR #{pr_number}...")
        status = self.get_pr_test_status(pr_number, token)
        if 'error' in status:
            self.logger.error(f"❌ Error getting status: {status['error']}")
            return status
        print("\n" + "="*70)
        print(f"📊 PR #{pr_number} TEST STATUS SUMMARY")
        print("="*70)
        print(f"✅ Passing: {status['total_passing']}")
        print(f"❌ Failing: {status['total_failing']}")
        if show_pending:
            print(f"⏳ Pending: {status['total_pending']}")
        print(f"🎯 AutoGluon Tests: {status['total_autogluon']}")
        if status['autogluon_tests']:
            print(f"\n🎯 AUTOGLUON TESTS ({len(status['autogluon_tests'])}):")
            for test in status['autogluon_tests']:
                emoji = "✅" if test['type'] == 'passing' else "❌" if test['type'] == 'failing' else "⏳"
                print(f"   {emoji} {test['name']} ({test['status']})")
        if status['failing']:
            print(f"\n❌ FAILING TESTS ({len(status['failing'])}):")
            for test in status['failing']:
                print(f"   - {test['name']} ({test['status']})")
        if show_pending and status['pending']:
            print(f"\n⏳ PENDING TESTS ({len(status['pending'])}):")
            for test in status['pending']:
                print(f"   - {test['name']} ({test['status']})")
        print("="*70)
        return status

    def wait_for_tests(self, pr_number: int = None, max_wait_minutes: int = 120, check_interval_minutes: int = 1) -> bool:
        """Wait for tests to complete and return True if all pass"""
        token = self.get_github_token()
        if not token:
            self.logger.error("❌ No GitHub token available for monitoring")
            return False
        if pr_number is None:
            existing_pr = self.check_existing_pr(token, include_closed=False)
            if existing_pr:
                pr_number = existing_pr['number']
            else:
                self.logger.error("❌ No PR found for current branch")
                return False
        self.logger.info(f"⏳ Waiting for tests to complete on PR #{pr_number}...")
        self.logger.info(f"Will check every {check_interval_minutes} minutes for up to {max_wait_minutes} minutes")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval_seconds = check_interval_minutes * 60
        test_history = []
        stable_count_needed = 3  
        while (time.time() - start_time) < max_wait_seconds:
            status = self.get_pr_test_status(pr_number, token)
            if 'error' in status:
                self.logger.error(f"❌ Error checking status: {status['error']}")
                return False
            total_tests = status['total_passing'] + status['total_failing'] + status['total_pending']
            test_history.append({
                'time': time.time(),
                'total': total_tests,
                'pending': status['total_pending'],
                'passing': status['total_passing'],
                'failing': status['total_failing'],
                'autogluon_total': status['total_autogluon']
            })
            test_history = test_history[-10:]
            if status['total_pending'] == 0 and total_tests > 0:
                recent_totals = [h['total'] for h in test_history[-stable_count_needed:]]
                recent_pending = [h['pending'] for h in test_history[-stable_count_needed:]]
                if (len(recent_totals) >= stable_count_needed and 
                    all(t == recent_totals[0] for t in recent_totals) and
                    all(p == 0 for p in recent_pending)):
                    self.logger.info(f"✅ Tests appear stable and complete")
                    if status['total_failing'] == 0:
                        self.logger.info(f"✅ All tests passed! ({status['total_passing']} passing, {status['total_autogluon']} AutoGluon)")
                        return True
                    else:
                        self.logger.info(f"❌ Tests completed with failures: {status['total_failing']} failing, {status['total_passing']} passing")
                        return False
                else:
                    self.logger.info(f"⏳ Tests appear complete but waiting for stability ({len(recent_totals)}/{stable_count_needed} stable checks)")
            else:
                status_parts = []
                if status['total_pending'] > 0:
                    status_parts.append(f"{status['total_pending']} pending")
                if status['total_passing'] > 0:
                    status_parts.append(f"{status['total_passing']} passing")
                if status['total_failing'] > 0:
                    status_parts.append(f"{status['total_failing']} failing")
                if status['total_autogluon'] > 0:
                    status_parts.append(f"{status['total_autogluon']} AutoGluon")
                self.logger.info(f"⏳ Tests still running: {', '.join(status_parts)}")
            time.sleep(check_interval_seconds)
        self.logger.warning(f"⏰ Timeout after {max_wait_minutes} minutes - tests still pending")
        return False

    def create_pull_request(self) -> bool:
        """Main method to create a pull request"""
        self.logger.info(f"🔄 Creating PR for AutoGluon {self.current_version}")
        if not self.push_branch_to_fork():
            return False
        token = self.get_github_token()
        if token:
            self.logger.info("🔑 Using GitHub token for API access")
            success = self.create_pr_via_api(token)
            self.logger.info("✅ PR creation completed successfully")
        else:
            self.logger.error("❌ PR creation failed")
        return success

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create PR for AutoGluon release')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.1)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    parser.add_argument('--repo-dir', required=True, help='Repository directory path')
    parser.add_argument('--max-wait', type=int, default=120, help='Maximum minutes to wait for tests (default: 120)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    pr_automation = GitHubPRAutomation(
        current_version=args.current_version,
        fork_url=args.fork_url,
        repo_dir=Path(args.repo_dir)
    )
    success = pr_automation.create_pull_request()
    if success:
        tests_passed = pr_automation.wait_for_tests(max_wait_minutes=args.max_wait)
        final_status = pr_automation.monitor_pr_status(show_pending=False)
        exit(0 if tests_passed else 1)
    else:
        exit(1)