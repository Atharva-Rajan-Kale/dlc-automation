import os
import re
import json
import logging
import subprocess
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import boto3
from datetime import datetime
import requests
import random
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from automation.common import BaseAutomation
from automation.automation_logger import LoggerMixin

class SecurityTestAgent(BaseAutomation,LoggerMixin):
    """Agentic system for automatically fixing security test failures"""
    
    def __init__(self, current_version: str, previous_version: str, fork_url: str):
        super().__init__(current_version, previous_version, fork_url)
        self.setup_github_credentials()
        self.setup_bedrock_client()
        self.setup_codebuild_client()
        self.setup_langchain()
        self.branch_name = f"autogluon-{current_version}-release"
        # Initialize container-specific tracking
        self.container_specific_logs = {
            'training': '',
            'inference': '',
            'unknown': ''
        }
        self.container_specific_vulnerabilities = {
            'training': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'inference': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'unknown': {'os_vulnerabilities': [], 'py_vulnerabilities': []}
        }
        self.setup_logging(current_version,custom_name="security_test")
        
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
        
    def setup_bedrock_client(self):
        """Initialize Bedrock client with dedicated AWS credentials"""
        self.logger.info("🔑 Setting up Bedrock client...")
        bedrock_access_key = os.getenv('BEDROCK_AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID')
        bedrock_secret_key = os.getenv('BEDROCK_AWS_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
        bedrock_region = os.getenv('BEDROCK_REGION') or os.getenv('REGION', 'us-east-1')
        bedrock_session_token = os.getenv('BEDROCK_AWS_SESSION_TOKEN') or os.getenv('AWS_SESSION_TOKEN')
        try:
            # Create Bedrock client with specific credentials
            if bedrock_access_key and bedrock_secret_key:
                self.logger.info("🎯 Using dedicated Bedrock credentials")
                session_kwargs = {
                    'aws_access_key_id': bedrock_access_key,
                    'aws_secret_access_key': bedrock_secret_key,
                    'region_name': bedrock_region
                }
                if bedrock_session_token:
                    session_kwargs['aws_session_token'] = bedrock_session_token
                session = boto3.Session(**session_kwargs)
                self.bedrock_client = session.client('bedrock-runtime')
            else:
                self.logger.info("🔧 Using default AWS credentials for Bedrock")
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=bedrock_region
                )
            self.logger.info("✅ Bedrock client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Bedrock client: {e}")
            self.logger.info("💡 Set BEDROCK_AWS_ACCESS_KEY_ID and BEDROCK_AWS_SECRET_ACCESS_KEY for dedicated Bedrock credentials")
            self.logger.info("💡 Or ensure default AWS credentials have Bedrock access")
            raise Exception("AWS credentials required for Bedrock access")
            
    def setup_codebuild_client(self):
        """Initialize CodeBuild client for accessing test logs"""
        self.logger.info("🔑 Setting up CodeBuild client for log access...")
        # Use specific environment variables for CodeBuild
        codebuild_access_key = os.getenv('CODEBUILD_AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID')
        codebuild_secret_key = os.getenv('CODEBUILD_AWS_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
        codebuild_region = os.getenv('CODEBUILD_REGION') or os.getenv('REGION', 'us-west-2')
        codebuild_session_token = os.getenv('CODEBUILD_AWS_SESSION_TOKEN') or os.getenv('AWS_SESSION_TOKEN')
        try:
            # Create CodeBuild client with specific credentials
            if codebuild_access_key and codebuild_secret_key:
                self.logger.info("🎯 Using dedicated CodeBuild credentials")
                session_kwargs = {
                    'aws_access_key_id': codebuild_access_key,
                    'aws_secret_access_key': codebuild_secret_key,
                    'region_name': codebuild_region
                }
                if codebuild_session_token:
                    session_kwargs['aws_session_token'] = codebuild_session_token
                session = boto3.Session(**session_kwargs)
                self.codebuild_client = session.client('codebuild')
                self.logs_client = session.client('logs')
            else:
                self.logger.info("🔧 Using default AWS credentials for CodeBuild")
                self.codebuild_client = boto3.client('codebuild', region_name=codebuild_region)
                self.logs_client = boto3.client('logs', region_name=codebuild_region)
            self.logger.info("✅ CodeBuild client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize CodeBuild client: {e}")
            self.logger.info("💡 Set CODEBUILD_AWS_ACCESS_KEY_ID and CODEBUILD_AWS_SECRET_ACCESS_KEY for dedicated CodeBuild credentials")
            raise Exception("AWS CodeBuild credentials required for log access")

    def setup_langchain(self):
        """Initialize LangChain with Claude via Bedrock - AI for detection"""
        model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        inference_profile_arn = os.getenv('BEDROCK_INFERENCE_PROFILE_ARN')
        if inference_profile_arn:
            self.logger.info(f"🎯 Using Bedrock inference profile: {inference_profile_arn}")
            try:
                self.llm = ChatBedrock(
                    client=self.bedrock_client,
                    model_id=inference_profile_arn,
                    provider="anthropic",
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                )
                self.logger.info("✅ Successfully initialized Bedrock with inference profile")
            except Exception as e:
                self.logger.error(f"❌ Failed with inference profile: {e}")
                self.logger.info("🔄 Falling back to regular model ID...")
                self.llm = ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model_id,
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                )
        else:
            self.logger.info(f"🎯 Using Bedrock model ID: {model_id}")
            self.llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=model_id,
                model_kwargs={
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            )        
        self.detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a vulnerability detection AI. Your ONLY job is to extract vulnerability information from security scan logs.

    ## WHAT TO EXTRACT
    ### OS Scan Vulnerabilities (ECR Enhanced Scan):
    Look for JSON patterns with vulnerability_id and package names:
    Total of X vulnerabilities need to be fixed on [container]:
    {{"package_name": [{{"vulnerability_id": "CVE-YYYY-NNNN", "description": "...", "severity": "..."}}]}}

    ### Python Scan Vulnerabilities (Safety Reports):
    Look for SAFETY_REPORT patterns in various formats:
    1. SAFETY_REPORT (FAILED) [pkg: package_name] [...] vulnerability_id='12345' [...] advisory='...'
    2. SafetyVulnerabilityAdvisory(vulnerability_id='12345', advisory="...", spec='...')
    3. [pkg: package_name] [installed: version] [vulnerabilities: [...]]

    Extract ALL Python vulnerabilities you find, including:
    - Package name from [pkg: ...] 
    - Vulnerability ID from vulnerability_id='...'
    - Full advisory text from advisory='...' or advisory="..."
    - Version constraints from spec='...'

    ## CRITICAL INSTRUCTIONS FOR PYTHON SCANS
    1. **FIND ALL SAFETY_REPORT LINES**: Look for any line containing "SAFETY_REPORT" and "(FAILED)"
    2. **EXTRACT COMPLETE ADVISORY**: Get the full advisory text, don't truncate it
    3. **HANDLE DIFFERENT QUOTE STYLES**: Advisory text may be in single quotes '...' or double quotes "..."
    4. **PRESERVE VERSION INFO**: If you see spec='<5.8.0' or similar, include this in the description
    5. **DEDUP BY PACKAGE+VULN_ID**: Same vulnerability_id in same package should only appear once
    6. **SCAN THOROUGHLY**: A single package can have MULTIPLE different vulnerabilities - extract ALL of them
    7. **DON'T STOP EARLY**: Continue scanning the entire log even after finding vulnerabilities

    ## SPECIAL ATTENTION FOR TRANSFORMERS PACKAGE
    The transformers package commonly has MULTIPLE vulnerabilities (77990, 77714, 77988, 77985, 77986, 78153).
    Make sure to extract ALL transformers vulnerabilities, not just the first one you find.

    ## OUTPUT FORMAT
    Return ONLY a simple JSON list of detected vulnerabilities:
    ```json
    {{
    "os_vulns": [
        {{
        "package": "package_name",
        "vulnerability_id": "CVE-YYYY-NNNN", 
        "description": "description text INCLUDING upgrade instructions",
        "severity": "CRITICAL|HIGH|MEDIUM|LOW"
        }}
    ],
    "py_vulns": [
        {{
        "package": "package_name",
        "vulnerability_id": "vulnerability_id",
        "description": "FULL advisory text INCLUDING upgrade instructions and spec info",
        "severity": "UNKNOWN",
        "spec": "version_constraint_if_found"
        }}
    ]
    }}

    Focus on accurate detection and complete data extraction. For Python scans, include ALL available information.
    BE ESPECIALLY THOROUGH - scan the ENTIRE log content for all vulnerabilities."""),
            ("human", """Extract all vulnerabilities from these security logs:
    Security Test Logs:
    {security_logs}

    Extract every vulnerability you can find - both OS scan CVEs and Python Safety reports. Pay special attention to Python SAFETY_REPORT lines and extract complete advisory information. 

    IMPORTANT: Make sure to find ALL vulnerabilities for each package, especially transformers which may have multiple vulnerabilities (77990, 77714, 77988, 77985, 77986, 78153, etc.).

    Return only the detection results in JSON format.""")
        ])
        self.detection_chain = self.detection_prompt | self.llm | JsonOutputParser()
    def _get_graphql_tests_for_commit(self, pr_number: int, commit_sha: str) -> List[Dict]:
        """Get tests for a specific commit via GraphQL with enhanced ID retrieval"""
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
                check_suites = commit_data.get('checkSuites', {}).get('nodes', [])
                for suite in check_suites:
                    suite_id = suite.get('id')
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
                        check_run_id = run.get('databaseId') or run.get('id')
                        tests.append({
                            'name': run['name'],
                            'state': state,
                            'type': 'check_run',
                            'check_run_id': check_run_id,
                            'url': run.get('url', ''),
                            'details_url': run.get('detailsUrl', ''),
                            'app': app_name,
                            'suite_id': suite_id
                        })
                        if check_run_id:
                            self.logger.info(f"🔍 GraphQL found check_run_id for {run['name']}: {check_run_id}")
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

    def _is_security_test(self, test_name: str) -> bool:
        """Enhanced security test detection with detailed logging"""
        test_name_lower = test_name.lower()
        if 'dlc-pr-quick-checks' in test_name_lower:
            return False
        security_indicators = [
            ('security', 'security' in test_name_lower),
            ('scan', 'scan' in test_name_lower),
            ('test_ecr_enhanced_scan', 'test_ecr_enhanced_scan' in test_name),
            ('test_safety_report_file', 'test_safety_report_file' in test_name)
        ]
        for indicator, matches in security_indicators:
            if matches:
                return True
        return False

    def is_package_in_allowlist(self, package_name: str, container_type: str, device_type: str) -> Dict[str, bool]:
        """Check if package exists in either OS or Python scan allowlists"""
        try:
            os_allowlist = self.get_allowlist_content(container_type, device_type, 'os_scan')
            py_allowlist = self.get_allowlist_content(container_type, device_type, 'py_scan')
            in_os_allowlist = package_name.lower() in os_allowlist
            in_py_allowlist = False
            for vuln_id, description in py_allowlist.items():
                if package_name.lower() in description.lower():
                    in_py_allowlist = True
                    break
            return {
                'os_allowlist': in_os_allowlist,
                'py_allowlist': in_py_allowlist,
                'any_allowlist': in_os_allowlist or in_py_allowlist
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check allowlists for {package_name}: {e}")
            return {'os_allowlist': False, 'py_allowlist': False, 'any_allowlist': False}
        
    def should_skip_dockerfile_fix(self, package_name: str, vuln_type: str) -> bool:
        """Determine if package should skip Dockerfile fixes and go straight to allowlist"""
        # Check torch package
        if package_name.lower() == 'torch':
            self.logger.info(f"🔄 Skipping Dockerfile for torch package: {package_name}")
            return True
        for container_type in ['training', 'inference']:
            for device_type in ['cpu', 'gpu']:
                allowlist_status = self.is_package_in_allowlist(package_name, container_type, device_type)
                if allowlist_status['any_allowlist']:
                    self.logger.info(f"🔄 Skipping Dockerfile for {package_name} - already in {container_type}/{device_type} allowlist")
                    return True
        return False
    
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

    def get_failing_security_tests(self, pr_number: int) -> List[Dict]:
        """Get failing security test information from PR"""
        try:
            all_security_tests = self.get_all_security_tests(pr_number)
            # Filter for only failing ones
            failing_security_tests = []
            for test in all_security_tests:
                # Check if test is failing
                is_failing = (
                    test.get('status') == 'failure' or 
                    test.get('conclusion') == 'failure' or
                    test.get('state') in ['FAILURE', 'ERROR', 'failure', 'error']
                )
                if is_failing:
                    failing_security_tests.append({
                        'name': test['name'],
                        'check_run_id': test.get('check_run_id'),
                        'url': test.get('url', ''),
                        'details_url': test.get('details_url', ''),
                        'source': test.get('source', 'unknown')
                    })
            self.logger.info(f"🔍 Found {len(failing_security_tests)} failing security tests")
            for test in failing_security_tests:
                self.logger.info(f"   - {test['name']} (source: {test.get('source', 'unknown')})")
            return failing_security_tests
        except Exception as e:
            self.logger.error(f"❌ Failed to get failing security tests: {e}")
            return []

    def get_logs_from_test_url(self, test_url: str, test_name: str) -> str:
        """Get logs by following the test URL programmatically"""
        if not test_url:
            return ""
        self.logger.info(f"🔗 Attempting to get logs from URL: {test_url}")
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            # First, get the test page
            response = requests.get(test_url, headers=headers, allow_redirects=True)
            response.raise_for_status()
            page_content = response.text
            self.logger.info(f"📄 Retrieved page content ({len(page_content)} chars)")
            # Check if this is a CodeBuild page
            if 'codebuild' in test_url or 'codebuild' in page_content:
                self.logger.info("🔍 Detected CodeBuild page, extracting logs...")
                return self.extract_logs_from_codebuild_page(page_content, test_url)
            # Check if this is a GitHub Actions page
            elif 'github.com' in test_url and 'actions' in test_url:
                self.logger.info("🔍 Detected GitHub Actions page, extracting logs...")
                return self.extract_logs_from_github_actions_page(page_content, test_url)
            # Try to find log content directly in the page
            else:
                self.logger.info("🔍 Searching for log content in page...")
                return self.extract_logs_from_generic_page(page_content)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get logs from URL {test_url}: {e}")
            return ""

    def extract_logs_from_codebuild_page(self, page_content: str, url: str) -> str:
        """Extract logs from CodeBuild console page"""
        try:
            # Extract build ID from URL or page content
            build_id_match = re.search(r'/codesuite/codebuild/projects/[^/]+/build/([^/?]+)(?:/log)?', url)
            if build_id_match:
                import urllib.parse
                raw_build_id = build_id_match.group(1)
                build_id = urllib.parse.unquote(raw_build_id)  # This converts %3A to :
                self.logger.info(f"📋 Found CodeBuild build ID: {build_id}")
                # Try to get logs via CodeBuild API
                try:
                    response = self.codebuild_client.batch_get_builds(ids=[build_id])
                    if response['builds']:
                        build = response['builds'][0]
                        if 'logs' in build and 'groupName' in build['logs']:
                            log_group = build['logs']['groupName']
                            log_stream = build['logs']['streamName']
                            
                            log_response = self.logs_client.get_log_events(
                                logGroupName=log_group,
                                logStreamName=log_stream
                            )
                            log_lines = [event['message'] for event in log_response['events']]
                            logs_content = '\n'.join(log_lines)
                            self.logger.info(f"✅ Retrieved {len(log_lines)} log lines from CodeBuild")
                            return logs_content
                except Exception as e:
                    self.logger.warning(f"⚠️ CodeBuild API access failed: {e}")
            log_patterns = [
                r'<pre[^>]*>(.*?)</pre>',
                r'<code[^>]*>(.*?)</code>',
                r'"logEvents":\s*\[(.*?)\]',
                r'log-content[^>]*>(.*?)</div>'
            ]
            for pattern in log_patterns:
                matches = re.findall(pattern, page_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    for match in matches:
                        clean_content = re.sub(r'<[^>]+>', '', match)
                        clean_content = clean_content.replace('&lt;', '<').replace('&gt;', '>')
                        clean_content = clean_content.replace('&amp;', '&').replace('&quot;', '"')
                        if len(clean_content) > 100:
                            self.logger.info(f"✅ Extracted logs from CodeBuild page ({len(clean_content)} chars)")
                            return clean_content
            return ""
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract CodeBuild logs: {e}")
            return ""
    
    def extract_logs_from_github_actions_page(self, page_content: str, url: str) -> str:
        """Extract logs from GitHub Actions page"""
        try:
            log_patterns = [
                r'data-log-text="([^"]*)"',
                r'<div class="log-line[^>]*>([^<]*)</div>',
                r'"message":"([^"]*)"',
                r'log-content[^>]*>([^<]*)</div>'
            ]
            all_log_content = []
            for pattern in log_patterns:
                matches = re.findall(pattern, page_content, re.IGNORECASE)
                all_log_content.extend(matches)
            if all_log_content:
                combined_logs = '\n'.join(all_log_content)
                combined_logs = combined_logs.replace('\\n', '\n').replace('\\t', '\t')
                self.logger.info(f"✅ Extracted GitHub Actions logs ({len(combined_logs)} chars)")
                return combined_logs            
            return ""
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract GitHub Actions logs: {e}")
            return ""

    def extract_logs_from_generic_page(self, page_content: str) -> str:
        """Extract log content from generic page"""
        try:
            patterns = [
                r'CVE-\d{4}-\d{4,}.*?(?=CVE-\d{4}-\d{4,}|$)',
                r'"vulnerability_id":\s*"[^"]*".*?(?="vulnerability_id"|$)',
                r'ERROR.*?(?=ERROR|$)',
                r'FAILED.*?(?=FAILED|$)',
                r'security.*?vulnerability.*?(?=security|$)'
            ]
            found_content = []
            for pattern in patterns:
                matches = re.findall(pattern, page_content, re.DOTALL | re.IGNORECASE)
                found_content.extend(matches)
            if found_content:
                combined_content = '\n'.join(found_content)
                self.logger.info(f"✅ Extracted security-related content ({len(combined_content)} chars)")
                return combined_content
            return ""
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract generic logs: {e}")
            return ""

    def get_allowlist_content(self, container_type: str, device_type: str, scan_type: str) -> Dict:
        """Get current allowlist content"""
        major_minor = '.'.join(self.current_version.split('.')[:2])
        if device_type == 'cpu':
            allowlist_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu.{scan_type}_allowlist.json"
        else:
            py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
            cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
            if cuda_dirs:
                allowlist_path = cuda_dirs[0] / f"Dockerfile.gpu.{scan_type}_allowlist.json"
            else:
                return {}
        if allowlist_path.exists():
            try:
                with open(allowlist_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def apply_allowlist_fixes(self, container_type: str, os_fixes: List[Dict], py_fixes: List[Dict]) -> bool:
        """Apply security fixes to allowlist files for specific container type only"""
        self.logger.info(f"🔧 Applying allowlist fixes for {container_type} only")
        success = True
        for device_type in ['cpu', 'gpu']:
            try:
                # Apply OS scan allowlist fixes (ECR Enhanced Scan format)
                if os_fixes:
                    current_os_allowlist = self.get_allowlist_content(container_type, device_type, 'os_scan')
                    self.logger.info("="*80)
                    self.logger.info(f"🔍 OS ALLOWLIST ENTRIES BEING ADDED ({container_type}/{device_type})")
                    self.logger.info("="*80)
                    for fix in os_fixes:
                        vuln_id = fix['vulnerability_id']
                        package = fix['package']
                        if 'original_vulnerability_data' in fix and fix['original_vulnerability_data']:
                            allowlist_entry = fix['original_vulnerability_data'].copy()
                            allowlist_entry['reason_to_ignore'] = "Security vulnerability allowlisted for AutoGluon DLC"
                            self.logger.info(f"📋 Using ORIGINAL data format for {vuln_id}:")
                            entry_keys = list(allowlist_entry.keys())[:5]
                            self.logger.info(f"   Field order: {entry_keys}...")
                        else:
                            allowlist_entry = {
                                "description": fix.get('description', f'Vulnerability {vuln_id}'),
                                "vulnerability_id": vuln_id,
                                "name": vuln_id,
                                "package_name": package,
                                "package_details": {
                                    "file_path": f"/opt/conda/lib/python3.11/site-packages/{package}",
                                    "name": package,
                                    "package_manager": "PYTHON",
                                    "version": "unknown",
                                    "release": None
                                },
                                "remediation": {
                                    "recommendation": {
                                        "text": "None Provided"
                                    }
                                },
                                "cvss_v3_score": 0.0,
                                "cvss_v30_score": 0.0,
                                "cvss_v31_score": 0.0,
                                "cvss_v2_score": 0.0,
                                "cvss_v3_severity": fix.get('severity', 'UNKNOWN'),
                                "source_url": f"https://nvd.nist.gov/vuln/detail/{vuln_id}",
                                "source": "NVD",
                                "severity": fix.get('severity', 'UNKNOWN'),
                                "status": "ACTIVE",
                                "title": f"{vuln_id} - {package}",
                                "reason_to_ignore": "Security vulnerability allowlisted for AutoGluon DLC"
                            }
                            self.logger.warning(f"⚠️ Using FALLBACK data for {vuln_id} with preserved field order")
                        # Group by package name
                        package_key = package.lower()
                        if package_key not in current_os_allowlist:
                            current_os_allowlist[package_key] = []
                        # Check if already exists
                        exists = any(item.get('vulnerability_id') == vuln_id for item in current_os_allowlist.get(package_key, []))
                        if not exists:
                            current_os_allowlist[package_key].append(allowlist_entry)
                            self.logger.info(f"✅ Added OS allowlist entry: {vuln_id} to '{package_key}' section in {container_type}")
                        else:
                            self.logger.info(f"⚠️ OS allowlist entry already exists: {vuln_id} in '{package_key}' for {container_type}")
                    self.logger.info("="*80)
                    self.save_allowlist(container_type, device_type, 'os_scan', current_os_allowlist)
                if py_fixes:
                    current_py_allowlist = self.get_allowlist_content(container_type, device_type, 'py_scan')
                    self.logger.info("="*80)
                    self.logger.info(f"🔍 PYTHON ALLOWLIST ENTRIES BEING ADDED ({container_type}/{device_type})")
                    self.logger.info("="*80)
                    for fix in py_fixes:
                        vuln_id = fix['vulnerability_id']
                        description = fix['description']
                        # Simple format: {"vulnerability_id": "description"}
                        if vuln_id not in current_py_allowlist:
                            current_py_allowlist[vuln_id] = description
                            self.logger.info(f"📝 Added PyScan allowlist entry to {container_type}:")
                            self.logger.info(f"  \"{vuln_id}\": \"{description}\"")
                        else:
                            self.logger.info(f"⚠️ PyScan allowlist entry already exists: '{vuln_id}' in {container_type}")
                    self.logger.info("="*80)
                    self.save_allowlist(container_type, device_type, 'py_scan', current_py_allowlist)
            except Exception as e:
                self.logger.error(f"❌ Failed to apply allowlist fixes for {container_type}/{device_type}: {e}")
                success = False
        return success

    def collect_security_logs(self, failing_tests: List[Dict]) -> str:
        """Collect logs from all failing security tests and store container-specific logs"""
        all_logs = ""
        # Initialize container-specific storage
        self.container_specific_logs = {
            'training': '',
            'inference': '',
            'unknown': ''
        }
        for test in failing_tests:
            self.logger.info(f"📥 Getting logs for {test['name']}")
            # Detect container type from test name
            container_type = self.detect_container_type_from_test_name(test['name'])
            self.logger.info(f"🔍 Detected container type: {container_type} for test {test['name']}")
            logs = ""
            if not logs and test.get('url'):
                logs = self.get_logs_from_test_url(test['url'], test['name'])
            if not logs and test.get('details_url'):
                logs = self.get_logs_from_test_url(test['details_url'], test['name'])
            if logs:
                # Add to both all_logs and container-specific logs
                all_logs += f"\n\n=== {test['name']} ({container_type}) ===\n" + logs
                self.container_specific_logs[container_type] += f"\n\n=== {test['name']} ===\n" + logs
                self.logger.info(f"✅ Retrieved logs from {test['name']} ({len(logs)} chars) - stored as {container_type}")
            else:
                self.logger.warning(f"⚠️ Could not retrieve logs for {test['name']}")
        # Log summary of container-specific logs
        for container_type, logs in self.container_specific_logs.items():
            if logs.strip():
                self.logger.info(f"📊 {container_type} logs: {len(logs)} characters")
        # Store logs for version constraint parsing
        self.current_security_logs = all_logs
        return all_logs
    
    def commit_and_push_changes(self, commit_message: str) -> bool:
        """Commit and push changes to the branch with user confirmation"""
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
            
            # Show a preview of the changes
            self.logger.info("\n📄 Preview of changes:")
            diff_preview = self.run_subprocess_with_logging(["git", "diff", "--stat"], capture_output=True, text=True)
            if diff_preview.stdout:
                for line in diff_preview.stdout.strip().split('\n'):
                    self.logger.info(f"   {line}")
            
            # Show the commit message
            self.logger.info(f"\n💬 Commit message: {commit_message}")
            
            # Configure Git with token before pushing
            token = self.get_github_token()
            if not token:
                self.logger.error("❌ No GitHub token available for push")
                return False
            
            if not self.configure_git_with_token(token):
                self.logger.error("❌ Failed to configure git with token")
                return False
            
            self.run_subprocess_with_logging(["git", "add", "."], check=True)
            self.run_subprocess_with_logging(["git", "commit", "-m", commit_message], check=True)
            self.run_subprocess_with_logging(["git", "push", "origin", self.branch_name], check=True)
            
            self.logger.info(f"✅ Successfully committed and pushed: {commit_message}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to commit and push: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def save_allowlist(self, container_type: str, device_type: str, scan_type: str, allowlist: Dict) -> bool:
        """Save allowlist to file"""
        try:
            major_minor = '.'.join(self.current_version.split('.')[:2])
            if device_type == 'cpu':
                allowlist_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu.{scan_type}_allowlist.json"
            else:
                py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
                cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
                if not cuda_dirs:
                    return False
                allowlist_path = cuda_dirs[0] / f"Dockerfile.gpu.{scan_type}_allowlist.json"
            allowlist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(allowlist_path, 'w') as f:
                json.dump(allowlist, f, indent=4)
            self.logger.info(f"✅ Saved {scan_type} allowlist: {allowlist_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save allowlist: {e}")
            return False

    def selectively_revert_dockerfile_packages(self, container_type: str, packages_to_revert: List[str]) -> bool:
        """
        Selectively remove specific packages from Dockerfiles while keeping other security fixes
        Args:
            container_type: 'training' or 'inference'
            packages_to_revert: List of package names to remove from Dockerfile
        """
        self.logger.info(f"🔄 Selectively reverting {len(packages_to_revert)} packages from {container_type} Dockerfiles")
        if not packages_to_revert:
            self.logger.info("ℹ️ No packages to revert")
            return True
        for pkg in packages_to_revert:
            self.logger.info(f"   🗑️ Reverting: {pkg}")
        major_minor = '.'.join(self.current_version.split('.')[:2])
        success = True
        for device_type in ['cpu', 'gpu']:
            try:
                if device_type == 'cpu':
                    dockerfile_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
                else:
                    py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
                    cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
                    if not cuda_dirs:
                        continue
                    dockerfile_path = cuda_dirs[0] / "Dockerfile.gpu"
                if not dockerfile_path.exists():
                    continue
                content = dockerfile_path.read_text()
                lines = content.split('\n')
                # Remove lines containing the packages to revert
                new_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    # Check if this line contains any package we need to revert
                    should_remove_line = False
                    for pkg in packages_to_revert:
                        # Handle both apt and pip package formats
                        if (f" {pkg}" in line or f" {pkg}=" in line or f" {pkg}==" in line or 
                            line.endswith(f" {pkg}") or line.endswith(f" {pkg} \\") or
                            f"    {pkg}" in line or f"    {pkg}=" in line or f"    {pkg}==" in line):
                            should_remove_line = True
                            self.logger.info(f"   🗑️ Removing line: {line}")
                            break
                    if should_remove_line:
                        # Skip this line and handle multiline continuations
                        if line.endswith(' \\'):
                            if new_lines and new_lines[-1].strip().endswith(' \\'):
                                next_i = i + 1
                                has_next_package = False
                                while next_i < len(lines):
                                    next_line = lines[next_i].strip()
                                    if not next_line or next_line.startswith('#'):
                                        next_i += 1
                                        continue
                                    # Check if next line has a package we're NOT removing
                                    next_has_package_to_keep = True
                                    for pkg in packages_to_revert:
                                        if (f" {pkg}" in next_line or f" {pkg}=" in next_line or 
                                            f"    {pkg}" in next_line):
                                            next_has_package_to_keep = False
                                            break
                                    
                                    if next_has_package_to_keep and ('apt-get' in next_line or 'pip install' in next_line or '    ' in next_line):
                                        has_next_package = True
                                    break
                                if not has_next_package:
                                    # Remove the continuation backslash from previous line
                                    if new_lines and new_lines[-1].strip().endswith(' \\'):
                                        new_lines[-1] = new_lines[-1].rstrip().rstrip('\\').rstrip()
                    else:
                        new_lines.append(lines[i])
                    i += 1
                # Clean up empty security fix sections
                final_lines = []
                skip_next_empty = False
                for i, line in enumerate(new_lines):
                    if line.strip() == "# Security fixes - OS packages" or line.strip() == "# Security fixes - Python packages":
                        # Check if there are actual packages after this comment
                        has_packages = False
                        for j in range(i + 1, min(i + 10, len(new_lines))):
                            next_line = new_lines[j].strip()
                            if next_line and not next_line.startswith('#') and ('RUN' in next_line or '    ' in next_line):
                                if any(pkg_name for pkg_name in ['apt-get', 'pip install'] if pkg_name in next_line):
                                    has_packages = True
                                    break
                                if '    ' in next_line and not next_line.startswith('&&') and not next_line.startswith('rm '):
                                    has_packages = True
                                    break
                        if has_packages:
                            final_lines.append(line)
                        else:
                            # Skip this comment and the next RUN line if it's empty
                            skip_next_empty = True
                    elif skip_next_empty and (line.strip().startswith('RUN apt-get') or line.strip().startswith('RUN pip install')):
                        # Check if this RUN line actually has packages
                        has_actual_packages = False
                        for j in range(i, min(i + 5, len(new_lines))):
                            check_line = new_lines[j]
                            if '    ' in check_line and not ('&&' in check_line or 'rm ' in check_line or 'clean' in check_line):
                                has_actual_packages = True
                                break
                        if has_actual_packages:
                            final_lines.append(line)
                            skip_next_empty = False
                        else:
                            # Skip empty RUN blocks
                            continue
                    else:
                        final_lines.append(line)
                        skip_next_empty = False
                # Write back the modified content
                new_content = '\n'.join(final_lines)
                dockerfile_path.write_text(new_content)
                self.logger.info(f"✅ Selectively reverted packages from {dockerfile_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to selectively revert packages from {device_type} Dockerfile: {e}")
                success = False
        return success

    def extract_version_from_description(self, description: str, package_name: str, vuln_data: Dict) -> str:
        """Extract version constraint from vulnerability description using pattern matching and AI"""
        try:
            self.logger.info(f"🔍 Analyzing description for version info for {package_name}")
            # First try rule-based pattern matching
            rule_based_constraint = self.extract_version_from_description_patterns(description, package_name)
            if rule_based_constraint and rule_based_constraint != "latest":
                return rule_based_constraint
            # If pattern matching fails, use AI for complex cases
            ai_constraint = self.extract_version_from_description_with_ai(description, package_name, vuln_data)
            if ai_constraint and ai_constraint != "latest":
                return ai_constraint
            return "latest"
        except Exception as e:
            return "latest"

    def extract_version_from_description_patterns(self, description: str, package_name: str) -> str:
        """Extract version constraint using strict patterns"""
        try:
            strict_upgrade_patterns = [
                rf'[Uu]sers should upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Uu]sers should upgrade to ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Uu]sers should upgrade to {re.escape(package_name)} version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)', 
                rf'[Uu]sers should upgrade to [A-Za-z\s]+ version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)', 
                rf'[Rr]ecommended to upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Mm]ust upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Pp]lease upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Ss]hould upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) to receive a fix',
                rf'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) to resolve',
                rf'upgrade to [A-Za-z\s]+ version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) or later',
            ]
            for i, pattern in enumerate(strict_upgrade_patterns):
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    return f">={version}"            
            strict_fixed_patterns = [
                rf'[Tt]his issue (?:has been|is) (?:patched|fixed|resolved) in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'(?:patched|fixed|resolved) in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Ff]ix (?:is )?available in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Pp]atch available in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'{re.escape(package_name)} ([0-9]+\.[0-9]+(?:\.[0-9]+)?) (?:fixes|resolves|patches) this',
            ]
            for i, pattern in enumerate(strict_fixed_patterns):
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    return f">={version}"
            return "latest"
        except Exception as e:
            return "latest"

    def extract_version_from_description_with_ai(self, description: str, package_name: str, vuln_data: Dict) -> str:
        """Use AI to extract version constraint from complex vulnerability descriptions - ENHANCED"""
        try:
            if not hasattr(self, 'llm'):
                self.logger.warning("⚠️ AI not available for version extraction")
                return "latest"
            # Get current version from vulnerability data
            current_version = vuln_data.get('package_details', {}).get('version', 'unknown')
            if not current_version or current_version == 'unknown':
                # Try to extract from original data
                if 'original_data' in vuln_data:
                    current_version = vuln_data['original_data'].get('package_details', {}).get('version', 'unknown')
            self.logger.info(f"🤖 Using AI to extract version from description for {package_name} (current: {current_version})")
            # Enhanced AI prompt for version extraction
            version_extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a security vulnerability analyzer. Your job is to extract version upgrade information from vulnerability descriptions.
    TASK: Extract the minimum safe version that fixes the vulnerability.
    RULES:
    1. Look for explicit upgrade instructions like:
    - "Users should upgrade to version X.X.X"
    - "Users should upgrade to [package] version X.X.X" 
    - "upgrade to version X.X.X to receive a fix"
    - "fixed in version X.X.X"
    - "patched in version X.X.X"
    - "this issue has been fixed in version X.X.X"
    2. Look for vulnerability scope descriptions that imply a fix version:
    - "prior to version X.X.X" means safe version is X.X.X
    - "before version X.X.X" means safe version is X.X.X  
    - "through version X.X.X" means safe version is X.X.X + 1 patch
    3. Be very careful to distinguish between:
    - DESCRIPTIVE text (just describing what's vulnerable)
    - PRESCRIPTIVE text (telling user what to do)
    4. Only extract versions from PRESCRIPTIVE upgrade instructions, not descriptive vulnerability scope.
    5. Focus on finding clear, actionable upgrade guidance.
    OUTPUT FORMAT:
    Return ONLY the version number (e.g., "5.8.0") or "none" if no clear upgrade instruction can be found.
    Do not include ">=" or other operators, just the version number.
    Do not include any explanation or additional text."""),
                ("human", """Package: {package_name}
    Current Version: {current_version}
    Vulnerability Description: {description}

    Extract the minimum safe version from upgrade instructions:""")
            ])     
            version_chain = version_extraction_prompt | self.llm
            ai_response = version_chain.invoke({
                "package_name": package_name,
                "current_version": current_version,
                "description": description
            })
            # Parse AI response
            ai_version = str(ai_response).strip().lower()
            self.logger.info(f"🤖 AI raw response: '{ai_version}'")
            if ai_version == "none" or not ai_version:
                self.logger.info(f"🤖 AI found no upgrade instructions for {package_name}")
                return "latest"
            # Validate the extracted version format
            if re.match(r'^[0-9]+\.[0-9]+(?:\.[0-9]+)?$', ai_version):
                self.logger.info(f"🤖 AI extracted valid upgrade version for {package_name}: {ai_version}")
                return f">={ai_version}"
            else:
                self.logger.warning(f"🤖 AI returned invalid version format for {package_name}: '{ai_version}'")
                return "latest"
        except Exception as e:
            self.logger.warning(f"⚠️ AI version extraction failed for {package_name}: {e}")
            return "latest"
    
    def extract_all_safety_vulnerabilities_from_logs(self, logs: str) -> List[Dict]:
        """Extract all Python vulnerabilities with regex patterns"""
        try:
            self.logger.info("🔍 Extracting all Safety vulnerabilities from new format")
            vulnerabilities = []
            seen_vulnerabilities = set()  # Track (package, vulnerability_id) pairs
            pattern1 = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\] \[installed: ([^\]]+)\] \[vulnerabilities: \[(.*?)\]\]'
            matches1 = re.finditer(pattern1, logs, re.DOTALL)
            for match in matches1:
                package_name = match.group(1).strip()
                installed_version = match.group(2).strip()
                vulnerabilities_text = match.group(3).strip()
                self.logger.info(f"📋 Processing Safety report for package: {package_name} (installed: {installed_version})")
                self.logger.info(f"📋 Vulnerabilities section preview: {vulnerabilities_text[:200]}...")
                
                # Improved advisory patterns that handle parentheses and various quote styles
                advisory_patterns = [
                    # Pattern A: Single quotes for advisory (most common in logs)
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'.*?advisory=\'(.*?)\'.*?spec=\'([^\']+)\'.*?\)',
                    # Pattern B: Double quotes for advisory  
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'.*?advisory="(.*?)".*?spec=\'([^\']+)\'.*?\)',
                    # Pattern C: Single quotes, no spec field
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'.*?advisory=\'(.*?)\'.*?\)',
                    # Pattern D: Double quotes, no spec field
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'.*?advisory="(.*?)".*?\)',
                    # Pattern E: More flexible order with single quotes
                    r'vulnerability_id=\'([^\']+)\'.*?advisory=\'(.*?)\'.*?spec=\'([^\']+)\'',
                    # Pattern F: More flexible order with double quotes
                    r'vulnerability_id=\'([^\']+)\'.*?advisory="(.*?)".*?spec=\'([^\']+)\'',
                ]
                
                found_vulnerability = False
                for pattern_idx, advisory_pattern in enumerate(advisory_patterns):
                    advisory_matches = re.finditer(advisory_pattern, vulnerabilities_text, re.DOTALL)
                    for advisory_match in advisory_matches:
                        try:
                            vuln_id = advisory_match.group(1).strip()
                            advisory_text = advisory_match.group(2).strip()
                            spec = advisory_match.group(3).strip() if len(advisory_match.groups()) >= 3 else 'unknown'
                            unique_key = (package_name, vuln_id)
                            if unique_key in seen_vulnerabilities:
                                self.logger.info(f"⚠️ DUPLICATE SKIPPED: {package_name} - {vuln_id} (already processed)")
                                continue
                            seen_vulnerabilities.add(unique_key)
                            vulnerabilities.append({
                                'package': package_name,
                                'vulnerability_id': vuln_id,
                                'description': advisory_text,
                                'installed_version': installed_version,
                                'spec': spec,
                                'severity': 'UNKNOWN'
                            })
                            self.logger.info(f"✅ NEW (pattern {pattern_idx+1}): {package_name} - {vuln_id} ({len(advisory_text)} chars)")
                            found_vulnerability = True
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error parsing advisory match for {package_name}: {e}")
                            continue
                
                if not found_vulnerability:
                    self.logger.warning(f"❌ NO VULNERABILITIES EXTRACTED for {package_name}")
                    self.logger.warning(f"❌ Raw vulnerabilities text: {vulnerabilities_text}")
                    debug_vuln_ids = re.findall(r'vulnerability_id=\'([^\']+)\'', vulnerabilities_text)
                    if debug_vuln_ids:
                        # Try a simpler extraction for debugging
                        simple_pattern = r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\''
                        simple_matches = re.findall(simple_pattern, vulnerabilities_text)
                    else:
                        self.logger.warning(f"DEBUG: No vulnerability_id patterns found at all")            
            if len(vulnerabilities) == 0:
                safety_lines = re.findall(r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\]', logs)
                all_vuln_ids = re.findall(r'vulnerability_id=\'([^\']+)\'', logs)
            return vulnerabilities
        except Exception as e:
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def extract_complete_advisory_content(self, package_name: str, vulnerability_id: str, logs: str) -> str:
        """Extract advisory content from new SafetyVulnerabilityAdvisory format"""
        try:
            safety_advisory_pattern = rf'SafetyVulnerabilityAdvisory\(vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory="([^"]+)"[^)]*\)'
            match = re.search(safety_advisory_pattern, logs, re.DOTALL)
            if match:
                advisory_content = match.group(1).strip()
                return advisory_content
            safety_advisory_pattern_single = rf'SafetyVulnerabilityAdvisory\(vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory=\'([^\']+)\'[^)]*\)'
            match = re.search(safety_advisory_pattern_single, logs, re.DOTALL)
            if match:
                advisory_content = match.group(1).strip()
                return advisory_content
            package_line_pattern = rf'SAFETY_REPORT \(FAILED\) \[pkg: {re.escape(package_name)}\].*?\[vulnerabilities: \[(.*?)\]\]'
            package_match = re.search(package_line_pattern, logs, re.DOTALL)
            if package_match:
                vulnerabilities_section = package_match.group(1)
                vuln_in_section_pattern = rf'SafetyVulnerabilityAdvisory\(vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory="([^"]+)"[^)]*\)'
                vuln_match = re.search(vuln_in_section_pattern, vulnerabilities_section, re.DOTALL)
                if vuln_match:
                    advisory_content = vuln_match.group(1).strip()
                    return advisory_content
            vuln_context_pattern = rf'vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory="([^"]+)"'
            match = re.search(vuln_context_pattern, logs, re.DOTALL)
            if match:
                advisory_content = match.group(1).strip()
                return advisory_content
            if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
                for pkg_name, pkg_vulns in self.original_vulnerability_data.items():
                    if isinstance(pkg_vulns, list):
                        for vuln in pkg_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                description = vuln.get('description', '')
                                if description:
                                    return description
            return ""
        except Exception as e:
            return ""

    def get_optimal_version_constraint(self, package_name: str, vuln_data: Dict, all_logs: str, vulnerability_type: str = 'os_scan') -> str:
        """Description extraction"""
        if vulnerability_type == 'py_scan':
            vulnerability_id = vuln_data.get('vulnerability_id', '')
            full_advisory = self.extract_complete_advisory_content(package_name, vulnerability_id, all_logs)
            if full_advisory:
                version_constraint = self.extract_version_from_description_patterns(full_advisory, package_name)
                if version_constraint and version_constraint != "latest":
                    extracted_version = version_constraint.replace('>=', '').replace('>', '').replace('<', '').replace('=', '')
                    return version_constraint
                else:
                    self.logger.info(f"No explicit version found in pyscan advisory")
            else:
                self.logger.info(f"Could not extract advisory content")
        else:
            if 'original_data' in vuln_data:
                original_desc = vuln_data['original_data'].get('description', '')
            else:
                self.logger.info(f"🔍 DEBUG: No 'original_data' in vuln_data")
            direct_desc = vuln_data.get('description', '')
            full_description = ""
            if 'original_data' in vuln_data and vuln_data['original_data']:
                full_description = vuln_data['original_data'].get('description', '')
            if not full_description:
                full_description = vuln_data.get('description', '')
            if full_description:
                version_constraint = self.extract_version_from_description_patterns(full_description, package_name)
                if version_constraint and version_constraint != "latest":
                    extracted_version = version_constraint.replace('>=', '').replace('>', '').replace('<', '').replace('=', '')
                    return version_constraint
                else:
                    self.logger.info(f"No explicit version found in OS scan description")
            else:
                self.logger.info(f"No OS scan description available")
        return "skip_dockerfile"

    def wait_for_security_tests_to_complete(self, pr_number: int, max_wait_minutes: int = 180) -> bool:
        """
        Wait for security tests to start and complete using both GraphQL + REST API.
        GraphQL for AutoGluon security tests (2 tests) + REST API for other security tests.
        """
        self.logger.info(f"⏳ Waiting for security tests to complete on PR #{pr_number} (using GraphQL + REST API)...")
        self.logger.info(f"Will wait up to {max_wait_minutes} minutes for security tests to finish")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval_seconds = 20
        consecutive_stable_checks = 0
        required_stable_checks = 3  # Need 3 consecutive stable results
        while (time.time() - start_time) < max_wait_seconds:
            try:
                all_security_tests = self.get_all_security_tests(pr_number)
                if not all_security_tests:
                    elapsed_minutes = int((time.time() - start_time) / 60)
                    self.logger.info(f"📊 No security tests found yet (after {elapsed_minutes}m) - tests may not have started")
                    time.sleep(check_interval_seconds)
                    continue
                # Count tests by status (using unified status format)
                pending_security_tests = [t for t in all_security_tests if t.get('status') in ['pending', 'in_progress', 'queued', 'requested'] or t.get('state') in ['PENDING', 'IN_PROGRESS', 'QUEUED', 'REQUESTED']]
                completed_security_tests = [t for t in all_security_tests if t.get('status') in ['success', 'failure', 'completed'] or t.get('state') in ['SUCCESS', 'FAILURE', 'CANCELLED', 'TIMED_OUT', 'ERROR']]
                failed_security_tests = [t for t in all_security_tests if t.get('status') == 'failure' or t.get('state') in ['FAILURE', 'ERROR', 'CANCELLED', 'TIMED_OUT']]
                elapsed_minutes = int((time.time() - start_time) / 60)
                self.logger.info(f"📊 Security test status via GraphQL + REST API (after {elapsed_minutes}m):")
                self.logger.info(f"   Total found: {len(all_security_tests)}")
                self.logger.info(f"   Pending/Running: {len(pending_security_tests)}")
                self.logger.info(f"   Completed: {len(completed_security_tests)}")
                self.logger.info(f"   Failed: {len(failed_security_tests)}")
                # Log individual test states for debugging
                for test in all_security_tests:
                    test_status = test.get('state', test.get('status', 'UNKNOWN'))
                    source = test.get('source', 'unknown')
                    self.logger.info(f"     - {test['name']}: {test_status} ({source})")
                # Check if all security tests have completed
                if len(pending_security_tests) == 0 and len(all_security_tests) > 0:
                    consecutive_stable_checks += 1
                    self.logger.info(f"✅ All security tests completed ({consecutive_stable_checks}/{required_stable_checks} stable checks)")
                    if consecutive_stable_checks >= required_stable_checks:
                        if len(failed_security_tests) == 0:
                            self.logger.info("🎉 All security tests passed!")
                            return True
                        else:
                            self.logger.info(f"⚠️ {len(failed_security_tests)} security tests failed - ready for analysis")
                            return True
                else:
                    consecutive_stable_checks = 0
                    if len(all_security_tests) == 0:
                        self.logger.info("⏳ Security tests haven't started yet...")
                    else:
                        self.logger.info("⏳ Some security tests still running...")
                time.sleep(check_interval_seconds)
            except Exception as e:
                self.logger.warning(f"⚠️ Error checking security test status: {e}")
                time.sleep(check_interval_seconds)
                continue
        self.logger.warning(f"⏰ Timeout after {max_wait_minutes} minutes - security tests may still be running")
        self.logger.info("🔄 Proceeding with current test status...")
        return True

    def get_all_security_tests(self, pr_number: int) -> List[Dict]:
        """Get all security-related tests (both passing and failing) for status monitoring - COMPREHENSIVE VERSION"""
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
            all_security_tests = []
            # 1. Check REST API Check Runs for HEAD
            check_runs_url = f"https://api.github.com/repos/aws/deep-learning-containers/commits/{head_sha}/check-runs"
            response = requests.get(check_runs_url, headers=headers, params={"per_page": 100})
            if response.status_code == 200:
                check_runs = response.json().get('check_runs', [])
                for run in check_runs:
                    run_name = run['name']
                    if self._is_security_test(run_name):
                        # Map GitHub status to our status
                        if run['status'] == 'completed':
                            if run['conclusion'] == 'success':
                                status = 'success'
                            elif run['conclusion'] in ['failure', 'cancelled', 'timed_out']:
                                status = 'failure'
                            else:
                                status = 'completed'
                        else:
                            status = run['status']  # pending, in_progress, queued, etc.
                        all_security_tests.append({
                            'name': run_name,
                            'check_run_id': run['id'],
                            'status': status,
                            'conclusion': run.get('conclusion'),
                            'url': run['html_url'],
                            'details_url': run.get('details_url', ''),
                            'source': 'REST_check_run'
                        })
            # 2. Check REST API Commit Statuses for HEAD
            status_url = f"https://api.github.com/repos/aws/deep-learning-containers/commits/{head_sha}/status"
            response = requests.get(status_url, headers=headers)
            if response.status_code == 200:
                status_data = response.json()
                statuses = status_data.get('statuses', [])
                for status in statuses:
                    status_name = status['context']
                    if self._is_security_test(status_name):
                        # Map commit status to our format
                        if status['state'] == 'success':
                            mapped_status = 'success'
                        elif status['state'] in ['failure', 'error']:
                            mapped_status = 'failure'
                        elif status['state'] == 'pending':
                            mapped_status = 'pending'
                        else:
                            mapped_status = status['state']
                        # Check if we already have this test from check runs
                        existing = any(t['name'] == status_name for t in all_security_tests)
                        if not existing:
                            all_security_tests.append({
                                'name': status_name,
                                'check_run_id': None,
                                'status': mapped_status,
                                'state': status['state'],
                                'url': status.get('target_url', ''),
                                'details_url': status.get('target_url', ''),
                                'source': 'REST_status'
                            })
            # 3. Check GraphQL for current HEAD (critical for finding security tests)
            graphql_tests = self._get_graphql_tests_for_commit(pr_number, head_sha)
            for test in graphql_tests:
                test_name = test['name']
                if self._is_security_test(test_name):
                    # Check if we already have this test
                    existing = any(t['name'] == test_name for t in all_security_tests)
                    if not existing:
                        # Map GraphQL status to our format
                        if test['state'] == 'SUCCESS':
                            mapped_status = 'success'
                        elif test['state'] in ['FAILURE', 'ERROR', 'CANCELLED', 'TIMED_OUT']:
                            mapped_status = 'failure'
                        elif test['state'] == 'PENDING':
                            mapped_status = 'pending'
                        else:
                            mapped_status = test['state'].lower()
                        all_security_tests.append({
                            'name': test_name,
                            'check_run_id': test.get('check_run_id'),  # Now includes check_run_id from GraphQL
                            'status': mapped_status,
                            'state': test['state'],
                            'url': test.get('url', ''),
                            'details_url': test.get('details_url', ''),
                            'source': 'GraphQL'
                        })
            if len(all_security_tests) > 0:
                sources = [t['source'] for t in all_security_tests]
                self.logger.info(f"📊 Found {len(all_security_tests)} security tests via: {', '.join(set(sources))}")
                for test in all_security_tests:
                    self.logger.info(f"   - {test['name']}: {test['status']} ({test['source']})")
            return all_security_tests
        except Exception as e:
            self.logger.error(f"❌ Failed to get all security tests: {e}")
            return []

    def run_security_test_agent(self) -> bool:
        """
        Main agent execution that handles any vulnerabilities
        """
        return self.run_deterministic_security_analysis()
    
    def preprocess_logs_for_ai(self, logs: str) -> str:
        """Preprocess logs and preserve original vulnerability data for exact allowlist formatting"""
        processed_logs = logs
        self.original_vulnerability_data = {}
        lines = logs.split('\n')
        processed_blocks = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'Total of' in line and 'vulnerabilities need to be fixed' in line and ':' in line:
                json_content = ""
                for j in range(i + 1, min(i + 20, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith('{'):
                        brace_count = 0
                        json_lines = []
                        for k in range(j, len(lines)):
                            check_line = lines[k]
                            json_lines.append(check_line)
                            brace_count += check_line.count('{') - check_line.count('}')
                            if brace_count == 0 and any(char in check_line for char in '}'):
                                json_content = '\n'.join(json_lines)
                                i = k 
                                break
                            if k - j > 50:
                                break
                        break
                if json_content:
                    try:
                        json_content = json_content.strip()
                        if '{' in json_content:
                            json_start = json_content.find('{')
                            json_content = json_content[json_start:]
                        parsed_json = json.loads(json_content)
                        for package_name, package_vulns in parsed_json.items():
                            if isinstance(package_vulns, list):
                                # If package already exists, extend the list instead of overwriting
                                if package_name in self.original_vulnerability_data:
                                    self.original_vulnerability_data[package_name].extend(package_vulns)
                                else:
                                    self.original_vulnerability_data[package_name] = package_vulns
                        processed_blocks += 1
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"❌ Could not parse JSON block: {e}")
            i += 1
        return processed_logs

    def apply_dockerfile_fixes(self, container_type: str, fixes: List[Dict]) -> Tuple[bool, Dict[str, str]]:
        """
        Apply security fixes to Dockerfiles by appending to existing AutoGluon installation
        """
        self.logger.info(f"🔧 Applying Dockerfile fixes for {container_type} by extending AutoGluon installation")
        if not fixes:
            self.logger.warning("⚠️ No fixes to apply")
            return True, {}
        major_minor = '.'.join(self.current_version.split('.')[:2])
        success = True
        vulnerability_mapping = {}
        total_modifications = 0
        for device_type in ['cpu', 'gpu']:
            try:
                # Determine Dockerfile path for the specific container type
                if device_type == 'cpu':
                    dockerfile_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
                else:
                    py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
                    cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
                    if not cuda_dirs:
                        self.logger.warning(f"⚠️ No CUDA directories found for {container_type} GPU")
                        continue
                    dockerfile_path = cuda_dirs[0] / "Dockerfile.gpu"
                self.logger.info(f"🔍 Processing Dockerfile: {dockerfile_path}")
                if not dockerfile_path.exists():
                    self.logger.error(f"❌ Dockerfile not found: {dockerfile_path}")
                    success = False
                    continue
                try:
                    original_content = dockerfile_path.read_text()
                    self.logger.info(f"📄 Read Dockerfile: {len(original_content)} characters")
                except Exception as e:
                    self.logger.error(f"❌ Failed to read Dockerfile {dockerfile_path}: {e}")
                    success = False
                    continue
                lines = original_content.split('\n')
                autogluon_line_index = None
                for i, line in enumerate(lines):
                    if "&& pip install --no-cache-dir -U autogluon==${AUTOGLUON_VERSION}" in line:
                        autogluon_line_index = i
                        self.logger.info(f"📍 Found AutoGluon installation at line {i}: {line.strip()}")
                        break
                if autogluon_line_index is None:
                    self.logger.error(f"❌ Could not find AutoGluon installation line in {dockerfile_path}")
                    success = False
                    continue
                pip_packages_dict = {}
                for fix in fixes:
                    if fix['type'] == 'update_package':
                        package = fix['package']
                        version_constraint = fix.get('version', 'latest')
                        install_method = fix.get('install_method', 'pip')
                        vulnerability_id = fix.get('vulnerability_id', '')
                        if vulnerability_id:
                            vulnerability_mapping[vulnerability_id] = package
                        if install_method == 'pip':
                            # Check if this package already exists in the Dockerfile
                            package_already_exists = False
                            for line in lines:
                                if f"pip install" in line and f'"{package}' in line:
                                    self.logger.info(f"⚠️ Package {package} already appears to be installed in Dockerfile")
                                    package_already_exists = True
                                    break
                            if not package_already_exists:
                                if package in pip_packages_dict:
                                    existing_constraint = pip_packages_dict[package]
                                    if existing_constraint == "upgrade" and version_constraint != 'latest':
                                        pip_packages_dict[package] = version_constraint
                                    elif existing_constraint != "upgrade" and version_constraint != 'latest':
                                        self.logger.info(f"⚠️ Multiple constraints for {package}: keeping {existing_constraint}")
                                else:
                                    pip_packages_dict[package] = version_constraint
                                self.logger.info(f"📝 Will add pip package: {package} (constraint: {version_constraint}) for {vulnerability_id}")
                            else:
                                self.logger.info(f"⏭️ Skipping {package} - already in Dockerfile")                
                pip_install_lines = []
                for package, version_constraint in pip_packages_dict.items():
                    if version_constraint and version_constraint != 'latest':
                        if any(op in version_constraint for op in ['>=', '<=', '==', '!=', '<', '>', ',']):
                            pip_install_lines.append(f'&& pip install --no-cache-dir "{package}{version_constraint}"')
                        else:
                            pip_install_lines.append(f'&& pip install --no-cache-dir "{package}=={version_constraint}"')
                    else:
                        pip_install_lines.append(f'&& pip install --no-cache-dir "{package}" --upgrade')
                        self.logger.warning(f"⚠️ Using --upgrade for {package} - could be risky!")
                if pip_install_lines:
                    self.logger.info(f"📝 Inserting {len(pip_install_lines)} pip install lines after AutoGluon installation")                    
                    insert_index = autogluon_line_index + 1                    
                    for j, pip_line in enumerate(pip_install_lines):
                        formatted_line = f" {pip_line} \\"
                        lines.insert(insert_index + j, formatted_line)
                    try:
                        new_content = '\n'.join(lines)
                        dockerfile_path.write_text(new_content)
                        verification_content = dockerfile_path.read_text()
                        if len(verification_content) > len(original_content):
                            self.logger.info(f"✅ Successfully modified {dockerfile_path}")
                            self.logger.info(f"📊 Content size: {len(original_content)} -> {len(verification_content)} chars")
                            total_modifications += 1
                            self.logger.info("📝 Added security packages after AutoGluon installation:")
                            for line in pip_install_lines:
                                self.logger.info(f"   + {line}")
                        else:
                            self.logger.error(f"❌ File size didn't increase after write - modification may have failed")
                            success = False
                    except Exception as e:
                        self.logger.error(f"❌ Failed to write to Dockerfile {dockerfile_path}: {e}")
                        success = False
                else:
                    self.logger.info("ℹ️ No pip packages to add to this Dockerfile")
            except Exception as e:
                self.logger.error(f"❌ Failed to update Dockerfile for {device_type}: {e}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                success = False
        self.logger.info(f"📊 Modified {total_modifications} Dockerfiles for {container_type}")
        return success, vulnerability_mapping

    def run_deterministic_security_analysis(self) -> bool:
        """
        AI-Enhanced security analysis with container-specific handling:
        1. Use AI to extract vulnerabilities by container type
        2. Try Dockerfile fixes first (container-specific)
        3. Commit and test
        4. If still failing, remove from Dockerfile and add to allowlist (container-specific)
        """
        self.logger.info("🤖 Starting AI-Enhanced Security Analysis with Container-Specific Handling...")
        try:
            # Get current PR number
            pr_number = self.get_current_pr_number()
            if not pr_number:
                self.logger.error("❌ No PR found, cannot access logs")
                return False
            
            self.check_autogluon_test_failures(pr_number)
            # Get failing security tests
            failing_tests = self.get_failing_security_tests(pr_number)
            if not failing_tests:
                self.logger.info("✅ No failing security tests found!")
                return True
            # Collect all logs with container-specific tracking
            all_logs = self.collect_security_logs(failing_tests)
            if not all_logs.strip():
                self.logger.warning("⚠️ No security logs retrieved")
                return False
            # Use AI to extract all vulnerabilities by container type
            vulnerabilities_by_container = self.extract_all_vulnerabilities_by_container(all_logs)
            # Check if we found any vulnerabilities
            total_vulns = sum(len(container_vulns['os_vulnerabilities']) + len(container_vulns['py_vulnerabilities']) 
                            for container_vulns in vulnerabilities_by_container.values())
            if total_vulns == 0:
                self.logger.warning("⚠️ No vulnerabilities detected in logs")
                return False
            # Log summary by container type
            for container_type, vulns in vulnerabilities_by_container.items():
                os_count = len(vulns['os_vulnerabilities'])
                py_count = len(vulns['py_vulnerabilities'])
                if os_count > 0 or py_count > 0:
                    self.logger.info(f"📊 {container_type}: {os_count} OS vulnerabilities, {py_count} Python vulnerabilities")
            # Try Dockerfile fixes first (container-specific)
            # Prepare combined vulnerabilities for processing but maintain container tracking
            combined_vulnerabilities = {
                'os_vulnerabilities': [],
                'py_vulnerabilities': []
            }
            for container_type, vulns in vulnerabilities_by_container.items():
                combined_vulnerabilities['os_vulnerabilities'].extend(vulns['os_vulnerabilities'])
                combined_vulnerabilities['py_vulnerabilities'].extend(vulns['py_vulnerabilities'])
            dockerfile_success = self.attempt_dockerfile_fixes_first_by_container(combined_vulnerabilities)
            if dockerfile_success:
                # Commit and test the Dockerfile changes
                commit_msg = f"AutoGluon {self.current_version}: Container-specific Dockerfile security fixes"
                if self.commit_and_push_changes(commit_msg):
                    self.logger.info("✅ Container-specific Dockerfile fixes committed, waiting for test results...")
                    # Wait for tests and check results
                    if self.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=40):
                        # Check what's still failing after Dockerfile attempt
                        remaining_failures = self.get_failing_security_tests(pr_number)
                        if not remaining_failures:
                            self.logger.info("🎉 All vulnerabilities fixed with container-specific Dockerfile changes!")
                            return True
                        else:
                            self.logger.info(f"⚠️ {len(remaining_failures)} tests still failing after container-specific Dockerfile fixes")
                            # Handle remaining failures, revert failed fixes and allowlist (container-specific)
                            return self.handle_remaining_vulnerabilities_with_container_specific_allowlist(
                                pr_number, vulnerabilities_by_container, remaining_failures)
                    else:
                        self.logger.warning("⚠️ Timeout waiting for Dockerfile fix tests")
                        return False
                else:
                    self.logger.error("❌ Failed to commit Dockerfile fixes")
                    return False
            else:
                # If Dockerfile fixes failed to apply, fall back to allowlist immediately
                self.logger.warning("⚠️ Dockerfile fixes failed to apply, using container-specific allowlist approach")
                return self.apply_direct_allowlist_fixes_by_container(vulnerabilities_by_container)
        except Exception as e:
            self.logger.error(f"❌ Deterministic security analysis failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        
    def apply_direct_allowlist_fixes_by_container(self, vulnerabilities_by_container: Dict) -> bool:
        """Apply allowlist fixes directly by container type when Dockerfile approach isn't possible"""
        self.logger.info("📝 Applying direct container-specific allowlist fixes...")
        success = True
        for container_type, vulns in vulnerabilities_by_container.items():
            if container_type == 'unknown':
                self.logger.warning(f"⚠️ Skipping unknown container type allowlist fixes")
                continue
            os_vulns = vulns['os_vulnerabilities']
            py_vulns = vulns['py_vulnerabilities']
            if not os_vulns and not py_vulns:
                self.logger.info(f"ℹ️ No vulnerabilities to allowlist for {container_type}")
                continue
            # Create allowlist fixes for this container type
            os_allowlist_fixes = []
            for vuln in os_vulns:
                os_allowlist_fixes.append({
                    'vulnerability_id': vuln['vulnerability_id'],
                    'package': vuln['package'],
                    'description': vuln['description'],
                    'original_vulnerability_data': vuln.get('original_data')
                })
            py_allowlist_fixes = []
            for vuln in py_vulns:
                py_allowlist_fixes.append({
                    'vulnerability_id': vuln['vulnerability_id'],
                    'description': vuln['description']
                })
            self.logger.info(f"📝 Applying {len(os_allowlist_fixes)} OS and {len(py_allowlist_fixes)} Python allowlist fixes to {container_type}")
            container_success = self.apply_allowlist_fixes(container_type, os_allowlist_fixes, py_allowlist_fixes)
            if not container_success:
                success = False
        if success:
            commit_msg = f"AutoGluon {self.current_version}: Apply container-specific security allowlist fixes"
            return self.commit_and_push_changes(commit_msg)
        return success
    
    def truncate_logs_for_ai(self, logs: str, max_chars: int = 100000) -> str:
        """Enhanced log truncation that preserves more vulnerability data"""
        if len(logs) <= max_chars:
            return logs        
        # Extract ALL SAFETY_REPORT lines first (these are critical)
        safety_lines = []
        for line in logs.split('\n'):
            if 'SAFETY_REPORT' in line and 'FAILED' in line:
                safety_lines.append(line)
        
        # Extract ALL complete vulnerability JSON blocks
        vulnerability_blocks = []
        lines = logs.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'Total of' in line and 'vulnerabilities need to be fixed' in line and ':' in line:
                # Look for JSON starting in the next few lines
                json_start_idx = None
                for j in range(i + 1, min(i + 20, len(lines))):
                    if lines[j].strip().startswith('{'):
                        json_start_idx = j
                        break
                
                if json_start_idx is not None:
                    # Extract the complete JSON block
                    brace_count = 0
                    json_lines = []
                    json_end_idx = json_start_idx
                    for k in range(json_start_idx, len(lines)):
                        line_content = lines[k]
                        json_lines.append(line_content)
                        # Count braces to find complete JSON
                        brace_count += line_content.count('{') - line_content.count('}')
                        # If we've closed all braces, we have complete JSON
                        if brace_count == 0 and any(char in line_content for char in '}'):
                            json_end_idx = k
                            break
                    
                    if brace_count == 0:
                        complete_block = [line] + json_lines
                        block_text = '\n'.join(complete_block)
                        vulnerability_blocks.append(block_text)
                        i = json_end_idx + 1
                        continue
            i += 1
        
        # Combine critical content with priority order
        critical_content = '\n'.join(vulnerability_blocks)
        if safety_lines:
            critical_content += '\n\n=== SAFETY REPORTS ===\n' + '\n'.join(safety_lines)
        
        critical_size = len(critical_content)
        remaining_space = max_chars - critical_size

        if critical_size > max_chars:
            self.logger.warning(f"⚠️ Critical vulnerability data ({critical_size} chars) exceeds max_chars ({max_chars})!")
            self.logger.warning("⚠️ Returning all critical data despite size limit")
            return critical_content
        
        # Add other important lines if there's space
        other_important_lines = []
        if remaining_space > 1000:
            for line in lines:
                # Skip lines already included in vulnerability blocks or safety lines
                skip_line = False
                for block in vulnerability_blocks:
                    if line.strip() in block:
                        skip_line = True
                        break
                if line in safety_lines:
                    skip_line = True
                
                if not skip_line and line.strip():
                    if any(keyword in line.lower() for keyword in [
                        'error', 'failed', 'exception', 'traceback', 
                        'test', 'assert', 'cve', 'vulnerability'
                    ]):
                        other_important_lines.append(line)
        
        final_content = critical_content
        if other_important_lines and remaining_space > 100:
            other_content = '\n'.join(other_important_lines)
            if len(other_content) <= remaining_space:
                final_content += '\n\n=== OTHER LOG CONTENT ===\n' + other_content
            else:
                truncated_other = other_content[:remaining_space-50] + '\n[... truncated ...]'
                final_content += '\n\n=== OTHER LOG CONTENT ===\n' + truncated_other
        final_size = len(final_content)        
        return final_content
    
    def detect_container_type_from_test_name(self, test_name: str) -> str:
        """Detect container type from test name"""
        test_name_lower = test_name.lower()
        if 'training' in test_name_lower:
            return 'training'
        elif 'inference' in test_name_lower:
            return 'inference'
        else:
            if any(pattern in test_name_lower for pattern in ['train', 'training-']):
                return 'training'
            elif any(pattern in test_name_lower for pattern in ['infer', 'inference-']):
                return 'inference'
            else:
                return 'unknown'

    def ai_detect_vulnerabilities_only_with_retry(self, logs: str, max_retries: int = 1) -> Dict:
        """Use AI only for vulnerability detection - Enhanced for Python scans"""
        self.logger.info("🤖 Using AI for BOTH OS and Python vulnerability detection...")
        
        for attempt in range(max_retries):
            try:
                # Enhanced truncation that preserves more vulnerability data
                truncated_logs = self.truncate_logs_for_ai(logs, max_chars=100000)  # Increased limit
                # Preprocess logs to help AI detection and store original OS data
                processed_logs = self.preprocess_logs_for_ai(truncated_logs)
                self.logger.info(f"🧠 AI attempt {attempt + 1}/{max_retries} - detecting ALL vulnerabilities...")
                ai_response = self.detection_chain.invoke({"security_logs": processed_logs})
                self.logger.info("="*80)
                self.logger.info(f"🔍 DEBUG: AI DETECTION RESPONSE (Attempt {attempt + 1})")
                self.logger.info("="*80)
                self.logger.info(f"Response type: {type(ai_response)}")
                self.logger.info(f"Response content: {ai_response}")
                self.logger.info("="*80)
                if isinstance(ai_response, dict):
                    detected = ai_response
                else:
                    # Try to parse if it's a string
                    try:
                        detected = json.loads(str(ai_response))
                    except json.JSONDecodeError:
                        self.logger.error(f"❌ AI returned invalid JSON for detection (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            self.logger.info("🔄 Retrying AI detection...")
                            time.sleep(2)  # Brief delay before retry
                            continue
                        else:
                            self.logger.error("❌ All AI attempts failed, using fallback")
                            return {'os_vulns': [], 'py_vulns': []}
                # Get detected vulnerabilities
                ai_os_vulns = detected.get('os_vulns', [])
                ai_py_vulns = detected.get('py_vulns', [])
                self.logger.info(f"🤖 AI detected {len(ai_os_vulns)} OS and {len(ai_py_vulns)} Python vulnerabilities")
                
                # Validate that we got reasonable results
                if len(ai_os_vulns) == 0 and len(ai_py_vulns) == 0:
                    if 'SAFETY_REPORT' in logs or 'vulnerability_id' in logs:
                        self.logger.warning(f"⚠️ AI found no vulnerabilities but logs contain vulnerability indicators (attempt {attempt + 1})")
                        if attempt < max_retries - 1:
                            self.logger.info("🔄 Retrying AI detection...")
                            time.sleep(2)
                            continue
                # Match AI detections with complete original data for OS scans
                enhanced_os_vulns = []
                enhanced_py_vulns = []
                # Process OS vulnerabilities - match with original data
                for ai_vuln in ai_os_vulns:
                    package = ai_vuln.get('package', '').lower()
                    vuln_id = ai_vuln.get('vulnerability_id', '')
                    # Find complete original vulnerability data
                    original_data = self.find_original_vulnerability_data(package, vuln_id)
                    if original_data:
                        enhanced_vuln = original_data.copy()
                        # Update with AI-detected information
                        enhanced_vuln.update({
                            'package': package,
                            'vulnerability_id': vuln_id,
                            'original_data': original_data  # Store for allowlist formatting
                        })
                        # Use AI description if it has upgrade info that original doesn't
                        ai_description = ai_vuln.get('description', '')
                        original_description = original_data.get('description', '')
                        if ('upgrade' in ai_description.lower() and 'version' in ai_description.lower() and
                            not ('upgrade' in original_description.lower() and 'version' in original_description.lower())):
                            enhanced_vuln['description'] = ai_description
                            self.logger.info(f"📋 Using AI description with upgrade info for {vuln_id}")
                        else:
                            # Keep original description
                            enhanced_vuln['description'] = original_description
                        enhanced_os_vulns.append(enhanced_vuln)
                    else:
                        # Fallback to AI detection only
                        enhanced_vuln = {
                            'package': package,
                            'vulnerability_id': vuln_id,
                            'description': ai_vuln.get('description', f'Vulnerability {vuln_id}'),
                            'severity': ai_vuln.get('severity', 'UNKNOWN'),
                            'original_data': {}
                        }
                        enhanced_os_vulns.append(enhanced_vuln)
                        self.logger.warning(f"⚠️ No original data found, using AI detection only for {vuln_id} in {package}")
                # Process Python vulnerabilities
                for ai_vuln in ai_py_vulns:
                    enhanced_py_vuln = {
                        'package': ai_vuln.get('package', ''),
                        'vulnerability_id': ai_vuln.get('vulnerability_id', ''),
                        'description': ai_vuln.get('description', ''),
                        'severity': ai_vuln.get('severity', 'UNKNOWN'),
                        'spec': ai_vuln.get('spec', '')
                    }
                    enhanced_py_vulns.append(enhanced_py_vuln)
                return {
                    'os_vulns': enhanced_os_vulns,      
                    'py_vulns': enhanced_py_vulns 
                }
            except Exception as e:
                self.logger.error(f"❌ AI detection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info("🔄 Retrying AI detection...")
                    time.sleep(2)
                    continue
                else:
                    self.logger.error("❌ All AI attempts failed")
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Last resort fallback to regex
                    self.logger.info("🔄 Using regex fallback for Python vulnerabilities...")
                    return self.ai_detection_with_regex_fallback(logs)
    
    def ai_detection_with_regex_fallback(self, logs: str) -> Dict:
        """Fallback when AI completely fails - use regex for Python only"""
        self.logger.info("🔄 AI failed completely, using regex fallback for Python vulnerabilities...")
        try:
            # Try to get OS vulnerabilities from stored original data
            os_vulns = []
            if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
                for package_name, package_vulns in self.original_vulnerability_data.items():
                    if isinstance(package_vulns, list):
                        for vuln in package_vulns:
                            if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                                os_vulns.append({
                                    'package': package_name,
                                    'vulnerability_id': vuln['vulnerability_id'],
                                    'description': vuln.get('description', f'Vulnerability {vuln["vulnerability_id"]}'),
                                    'severity': vuln.get('severity', vuln.get('cvss_v3_severity', 'UNKNOWN')),
                                    'original_data': vuln
                                })
            # Use regex for Python vulnerabilities as last resort
            py_vulns = []
            py_vulnerabilities = self.extract_all_safety_vulnerabilities_from_logs(logs)
            for vuln in py_vulnerabilities:
                py_vulns.append({
                    'package': vuln['package'],
                    'vulnerability_id': vuln['vulnerability_id'],
                    'description': vuln['description'],
                    'severity': vuln.get('severity', 'UNKNOWN'),
                    'spec': vuln.get('spec', '')
                })
            self.logger.info(f"🔄 Fallback extracted {len(os_vulns)} OS and {len(py_vulns)} Python vulnerabilities")
            return {
                'os_vulns': os_vulns,
                'py_vulns': py_vulns
            }
        except Exception as e:
            self.logger.error(f"❌ Even regex fallback failed: {e}")
            return {'os_vulns': [], 'py_vulns': []}
        
    def extract_all_vulnerabilities_fallback(self, logs: str) -> Dict:
        """Fallback rule-based extraction when AI fails"""
        self.logger.info("🔍 Using fallback vulnerability extraction...")
        # First try AI detection one more time with simpler processing
        try:
            self.logger.info("🤖 Attempting simplified AI detection as fallback...")
            simplified_ai_result = self.ai_detect_vulnerabilities_only_with_retry(logs, max_retries=1)
            
            if (simplified_ai_result.get('os_vulns') or simplified_ai_result.get('py_vulns')):
                self.logger.info("✅ Simplified AI detection worked in fallback")
                return {
                    'os_vulnerabilities': simplified_ai_result.get('os_vulns', []),
                    'py_vulnerabilities': simplified_ai_result.get('py_vulns', [])
                }
        except Exception as e:
            self.logger.warning(f"⚠️ Simplified AI detection also failed: {e}")
        # If AI still fails, use the original regex-based approach
        self.logger.info("🔍 Using rule-based regex extraction as final fallback...")
        # Use preprocessing to extract vulnerability data
        processed_logs = self.preprocess_logs_for_ai(logs)
        os_vulnerabilities = []
        py_vulnerabilities = []
        if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
            self.logger.info(f"📊 Processing {len(self.original_vulnerability_data)} packages from OS scan")
            for package_name, package_vulns in self.original_vulnerability_data.items():
                if isinstance(package_vulns, list):
                    for vuln in package_vulns:
                        if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                            os_vulnerabilities.append({
                                'package': package_name,
                                'vulnerability_id': vuln['vulnerability_id'],
                                'description': vuln.get('description', f'Vulnerability {vuln["vulnerability_id"]}'),
                                'severity': vuln.get('severity', vuln.get('cvss_v3_severity', 'UNKNOWN')),
                                'original_data': vuln
                            })
                            self.logger.info(f"📋 Fallback found OS vulnerability: {vuln['vulnerability_id']} in {package_name}")
        safety_vulnerabilities = self.extract_all_safety_vulnerabilities_from_logs(logs)
        for vuln in safety_vulnerabilities:
            py_vulnerabilities.append({
                'package': vuln['package'],
                'vulnerability_id': vuln['vulnerability_id'],
                'description': vuln['description'],
                'severity': vuln.get('severity', 'UNKNOWN')
            })
            self.logger.info(f"📋 Fallback found Python vulnerability: {vuln['vulnerability_id']} in {vuln['package']}")
        self.logger.info(f"📊 Final fallback extracted {len(os_vulnerabilities)} OS and {len(py_vulnerabilities)} Python vulnerabilities")
        return {
            'os_vulnerabilities': os_vulnerabilities,
            'py_vulnerabilities': py_vulnerabilities
        }
    
    def ai_detect_vulnerabilities_only(self, logs: str) -> Dict:
        """Main entry point, delegates to retry version"""
        return self.ai_detect_vulnerabilities_only_with_retry(logs, max_retries=1)
        
    def extract_all_vulnerabilities(self, logs: str) -> Dict:
        """Extract all vulnerabilities using AI for detection only, then rule-based formatting"""
        self.logger.info("🤖 Using AI only for vulnerability detection, preserving complete original data...")
        try:
            detected_vulnerabilities = self.ai_detect_vulnerabilities_only(logs)            
            os_vulnerabilities = []
            py_vulnerabilities = []
            # Print detected vulnerabilities for debugging
            self.logger.info("="*80)
            self.logger.info("🔍 DEBUG: AI DETECTED VULNERABILITIES")
            self.logger.info("="*80)
            self.logger.info(f"OS vulnerabilities detected: {len(detected_vulnerabilities.get('os_vulns', []))}")
            for vuln in detected_vulnerabilities.get('os_vulns', []):
                self.logger.info(f"  - {vuln.get('vulnerability_id', 'Unknown')} in {vuln.get('package', 'Unknown')}")
            self.logger.info(f"Python vulnerabilities detected: {len(detected_vulnerabilities.get('py_vulns', []))}")
            for vuln in detected_vulnerabilities.get('py_vulns', []):
                self.logger.info(f"  - {vuln.get('vulnerability_id', 'Unknown')} in {vuln.get('package', 'Unknown')}")
            self.logger.info("="*80)
            for detected_vuln in detected_vulnerabilities.get('os_vulns', []):
                package = detected_vuln.get('package', '').lower()
                vuln_id = detected_vuln.get('vulnerability_id', '')
                original_data = self.find_original_vulnerability_data(package, vuln_id)
                if original_data:
                    complete_vuln = original_data.copy()                    
                    ai_description = detected_vuln.get('description', '')
                    original_description = original_data.get('description', '')
                    if ('upgrade' in ai_description.lower() and 'version' in ai_description.lower() and
                        not ('upgrade' in original_description.lower() and 'version' in original_description.lower())):
                        complete_vuln['description'] = ai_description
                        self.logger.info(f"📋 Using AI description with upgrade info for {vuln_id}")
                    complete_vuln.update({
                        'package': package,
                        'vulnerability_id': vuln_id,
                        'original_data': original_data  # Store complete original for allowlist formatting
                    })
                    os_vulnerabilities.append(complete_vuln)
                    self.logger.info(f"✅ Matched OS vulnerability with COMPLETE original data: {vuln_id} in {package}")
                else:
                    fallback_vuln = {
                        'package': package,
                        'vulnerability_id': vuln_id,
                        'description': detected_vuln.get('description', f'Vulnerability {vuln_id}'),
                        'severity': detected_vuln.get('severity', 'UNKNOWN'),
                        'original_data': {}
                    }
                    os_vulnerabilities.append(fallback_vuln)
                    self.logger.warning(f"⚠️ Using fallback data for OS vulnerability: {vuln_id} in {package}")
            # Handle Python vulnerabilities
            for detected_vuln in detected_vulnerabilities.get('py_vulns', []):
                py_vulnerabilities.append({
                    'package': detected_vuln.get('package', ''),
                    'vulnerability_id': detected_vuln.get('vulnerability_id', ''),
                    'description': detected_vuln.get('description', ''),
                    'severity': detected_vuln.get('severity', 'UNKNOWN')
                })
                self.logger.info(f"✅ Added Python vulnerability: {detected_vuln.get('vulnerability_id')} in {detected_vuln.get('package')}")
            self.logger.info(f"📊 Final result: {len(os_vulnerabilities)} OS and {len(py_vulnerabilities)} Python vulnerabilities with COMPLETE original data preserved")
            return {
                'os_vulnerabilities': os_vulnerabilities,
                'py_vulnerabilities': py_vulnerabilities
            }
        except Exception as e:
            self.logger.error(f"❌ Vulnerability extraction failed: {e}")
            self.logger.info("🔄 Falling back to rule-based extraction...")
            return self.extract_all_vulnerabilities_fallback(logs)
            
    def find_original_vulnerability_data(self, package_name: str, vulnerability_id: str) -> Dict:
        """Find original vulnerability data from preprocessed logs"""
        try:
            if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
                package_key = package_name.lower()
                # Look for the package in original data
                if package_key in self.original_vulnerability_data:
                    package_vulns = self.original_vulnerability_data[package_key]
                    if isinstance(package_vulns, list):
                        for vuln in package_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                return vuln 
                for stored_package, stored_vulns in self.original_vulnerability_data.items():
                    if isinstance(stored_vulns, list):
                        for vuln in stored_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                if (package_name.lower() in stored_package.lower() or 
                                    stored_package.lower() in package_name.lower()):
                                    return vuln
                return {}
            else:
                self.logger.warning(f"⚠️ No original_vulnerability_data available")
                return {}
        except Exception as e:
            self.logger.error(f"❌ Error finding original vulnerability data: {e}")
            return {}
        
    def extract_all_vulnerabilities_by_container(self, all_logs: str) -> Dict:
        """Extract vulnerabilities organized by container type"""
        self.logger.info("🤖 Extracting vulnerabilities by container type (allowing per-container duplicates)...")
        result = {
            'training': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'inference': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'unknown': {'os_vulnerabilities': [], 'py_vulnerabilities': []}
        }
        # Process each container type's logs separately
        container_order = ['training', 'inference', 'unknown']
        for i, container_type in enumerate(container_order):
            logs = self.container_specific_logs.get(container_type, '')
            if not logs.strip():
                continue
            self.logger.info(f"📊 Processing {container_type} container logs ({len(logs)} chars)")
            # Add 5-minute delay before inference processing to avoid throttling
            if container_type == 'inference' and i > 0:  # Only delay if not the first container
                self.logger.info("⏳ Adding 5-minute delay before inference AI processing to avoid throttling...")
                delay_seconds = 5 * 60
                for remaining in range(delay_seconds, 0, -30):
                    minutes = remaining // 60
                    seconds = remaining % 60
                    self.logger.info(f"⏳ Waiting {minutes}:{seconds:02d} before inference processing...")
                    time.sleep(30)
                self.logger.info("✅ Delay complete, proceeding with inference AI processing...")
            container_seen_os = set()  # Track (package, vuln_id) for this container only
            container_seen_py = set()  # Track (package, vuln_id) for this container only
            try:
                container_vulns = self.ai_detect_vulnerabilities_only_with_retry(logs)
                for vuln in container_vulns.get('os_vulns', []):
                    vuln['container_type'] = container_type
                    unique_key = (vuln['package'], vuln['vulnerability_id'])
                    if unique_key not in container_seen_os:
                        container_seen_os.add(unique_key)
                        result[container_type]['os_vulnerabilities'].append(vuln)
                        self.logger.info(f"✅ NEW OS ({container_type}): {vuln['package']} - {vuln['vulnerability_id']}")
                    else:
                        self.logger.info(f"⚠️ DUPLICATE OS WITHIN {container_type}: {vuln['package']} - {vuln['vulnerability_id']}")
                for vuln in container_vulns.get('py_vulns', []):
                    vuln['container_type'] = container_type
                    unique_key = (vuln['package'], vuln['vulnerability_id'])
                    if unique_key not in container_seen_py:
                        container_seen_py.add(unique_key)
                        result[container_type]['py_vulnerabilities'].append(vuln)
                        self.logger.info(f"✅ NEW PY ({container_type}): {vuln['package']} - {vuln['vulnerability_id']}")
                    else:
                        self.logger.info(f"⚠️ DUPLICATE PY WITHIN {container_type}: {vuln['package']} - {vuln['vulnerability_id']}")
            except Exception as e:
                self.logger.error(f"❌ Failed to extract vulnerabilities for {container_type}: {e}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
        self.container_specific_vulnerabilities = result
        total_os = sum(len(container['os_vulnerabilities']) for container in result.values())
        total_py = sum(len(container['py_vulnerabilities']) for container in result.values())
        return result

    def extract_all_vulnerabilities_fallback(self, logs: str) -> Dict:
        """Fallback rule-based extraction when AI fails"""
        self.logger.info("🔍 Using fallback rule-based vulnerability extraction...")
        processed_logs = self.preprocess_logs_for_ai(logs)
        os_vulnerabilities = []
        py_vulnerabilities = []
        if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
            self.logger.info(f"📊 Processing {len(self.original_vulnerability_data)} packages from OS scan")
            for package_name, package_vulns in self.original_vulnerability_data.items():
                if isinstance(package_vulns, list):
                    for vuln in package_vulns:
                        if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                            os_vulnerabilities.append({
                                'package': package_name,
                                'vulnerability_id': vuln['vulnerability_id'],
                                'description': vuln.get('description', f'Vulnerability {vuln["vulnerability_id"]}'),
                                'severity': vuln.get('severity', vuln.get('cvss_v3_severity', 'UNKNOWN')),
                                'original_data': vuln
                            })
                            self.logger.info(f"📋 Fallback found OS vulnerability: {vuln['vulnerability_id']} in {package_name}")
        safety_pattern = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'(?:.*?advisory=\'([^\']*?)\')?'        
        safety_matches = list(re.finditer(safety_pattern, logs, re.DOTALL))
        seen_py_vulns = set()
        for match in safety_matches:
            groups = match.groups()
            package = groups[0].strip()
            vuln_id = groups[1].strip()
            advisory = groups[2].strip() if groups[2] else f'Security vulnerability in {package}'
            self.logger.info(f"🔍 FALLBACK: Extracted advisory for {package} ({vuln_id}): '{advisory[:100]}{'...' if len(advisory) > 100 else ''}'")
            unique_key = (package, vuln_id)
            if unique_key not in seen_py_vulns:
                seen_py_vulns.add(unique_key)
                py_vulnerabilities.append({
                    'package': package,
                    'vulnerability_id': vuln_id,
                    'description': advisory,
                    'severity': 'UNKNOWN'
                })
                self.logger.info(f"📋 Fallback found Python vulnerability: {vuln_id} in {package}")
        self.logger.info(f"📊 Fallback extracted {len(os_vulnerabilities)} OS and {len(py_vulnerabilities)} Python vulnerabilities")
        return {
            'os_vulnerabilities': os_vulnerabilities,
            'py_vulnerabilities': py_vulnerabilities
        }

    def get_optimal_version_constraint_with_upgrade_check(self, package_name: str, vuln_data: Dict, all_logs: str, vulnerability_type: str = 'os_scan') -> str:
        """Simplified version with upgrade check, just calls the main constraint method"""
        constraint = self.get_optimal_version_constraint(package_name, vuln_data, all_logs, vulnerability_type)
        return constraint

    def check_pyscan_constraint_and_cross_contaminate(self, vulnerabilities: Dict, all_logs: str) -> Dict:
        """Check pyscan constraints and cross-contaminate to OS scan if needed"""
        failed_pyscan_packages = set()
        # Check all pyscan vulnerabilities first
        for vuln in vulnerabilities['py_vulnerabilities']:
            package = vuln['package']
            # Get pyscan constraint
            pyscan_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                package, {}, all_logs, 'py_scan'
            )
            if pyscan_constraint == "skip_dockerfile":
                failed_pyscan_packages.add(package)
        # Cross-contaminate: if pyscan failed, also allowlist any OS scan vulnerabilities for the same package
        if failed_pyscan_packages:
            self.logger.info(f"🔄 Cross-contaminating {len(failed_pyscan_packages)} packages: {failed_pyscan_packages}")
            for vuln in vulnerabilities['os_vulnerabilities']:
                if vuln['package'] in failed_pyscan_packages:
                    vuln['cross_contaminated'] = True
        return vulnerabilities

    def attempt_dockerfile_fixes_first_by_container(self, vulnerabilities: Dict) -> bool:
        """Apply Dockerfile fixes only to relevant container types"""
        all_logs = getattr(self, 'current_security_logs', '')
        vulnerabilities = self.check_pyscan_constraint_and_cross_contaminate(vulnerabilities, all_logs)
        # Group vulnerabilities by container type
        container_fixes = {
            'training': {'dockerfile_fixes': [], 'os_allowlist': [], 'py_allowlist': []},
            'inference': {'dockerfile_fixes': [], 'os_allowlist': [], 'py_allowlist': []},
            'unknown': {'dockerfile_fixes': [], 'os_allowlist': [], 'py_allowlist': []}
        }
        # Process OS vulnerabilities
        for vuln in vulnerabilities['os_vulnerabilities']:
            package = vuln['package']
            vuln_id = vuln['vulnerability_id']
            container_type = vuln.get('container_type', 'unknown')
            # Check if cross-contaminated (should go to allowlist due to pyscan failure)
            if vuln.get('cross_contaminated', False):
                container_fixes[container_type]['os_allowlist'].append({
                    'vulnerability_id': vuln_id,
                    'package': package,
                    'description': vuln['description'],
                    'original_vulnerability_data': vuln.get('original_data')
                })
                self.logger.info(f"📝 Cross-contaminated OS vulnerability to {container_type} allowlist: {vuln_id} in {package}")
                continue
            # Normal OS scan processing
            if self.should_skip_dockerfile_fix(package, 'os_scan'):
                container_fixes[container_type]['os_allowlist'].append({
                    'vulnerability_id': vuln_id,
                    'package': package,
                    'description': vuln['description'],
                    'original_vulnerability_data': vuln.get('original_data')
                })
                self.logger.info(f"📝 Pre-filtered to {container_type} OS allowlist: {vuln_id} in {package}")
            else:
                vuln_data_to_pass = vuln.get('original_data', {})
                if not vuln_data_to_pass:
                    vuln_data_to_pass = {
                        'description': vuln.get('description', ''),
                        'vulnerability_id': vuln_id,
                        'package_name': package,
                        'severity': vuln.get('severity', 'UNKNOWN')
                    }
                    self.logger.info(f"🔍 Created vuln_data for {package} from AI detection: {len(vuln_data_to_pass.get('description', ''))} chars")
                # Get optimal version constraint for OS scan 
                version_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                    package, 
                    vuln_data_to_pass,
                    all_logs,
                    'os_scan'
                )
                if version_constraint == "skip_dockerfile":
                    container_fixes[container_type]['os_allowlist'].append({
                        'vulnerability_id': vuln_id,
                        'package': package,
                        'description': vuln['description'],
                        'original_vulnerability_data': vuln.get('original_data')
                    })
                    self.logger.info(f"📝 No upgrade possible for {package} (OS scan), adding to {container_type} OS allowlist: {vuln_id}")
                else:
                    install_method = 'pip' if any(mgr in vuln.get('original_data', {}).get('package_details', {}).get('package_manager', 'PYTHON').upper() 
                                            for mgr in ['PYTHON', 'PIP']) else 'apt'
                    container_fixes[container_type]['dockerfile_fixes'].append({
                        'type': 'update_package',
                        'package': package,
                        'version': version_constraint,
                        'install_method': install_method,
                        'vulnerability_id': vuln_id,
                        'container_type': container_type,
                        'reasoning': f"OS scan fix for {vuln_id} with constraint {version_constraint}"
                    })
                    self.logger.info(f"📝 Will attempt {container_type} Dockerfile fix: {package} {version_constraint} via {install_method} for {vuln_id} (OS scan)")
        # Process Python vulnerabilities
        for vuln in vulnerabilities['py_vulnerabilities']:
            package = vuln['package']
            vuln_id = vuln['vulnerability_id']
            container_type = vuln.get('container_type', 'unknown')
            if self.should_skip_dockerfile_fix(package, 'py_scan'):
                container_fixes[container_type]['py_allowlist'].append({
                    'vulnerability_id': vuln_id,
                    'description': vuln['description']
                })
                self.logger.info(f"📝 Pre-filtered to {container_type} Python allowlist: {vuln_id} in {package}")
            else:
                # Get optimal version constraint for pyscan
                version_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                    package, 
                    vuln, 
                    all_logs,
                    'py_scan'
                )
                if version_constraint == "skip_dockerfile":
                    container_fixes[container_type]['py_allowlist'].append({
                        'vulnerability_id': vuln_id,
                        'description': vuln['description']
                    })
                    self.logger.info(f"📝 No upgrade possible for {package} (pyscan), adding to {container_type} Python allowlist: {vuln_id}")
                else:
                    container_fixes[container_type]['dockerfile_fixes'].append({
                        'type': 'update_package',
                        'package': package,
                        'version': version_constraint,
                        'install_method': 'pip',
                        'vulnerability_id': vuln_id,
                        'container_type': container_type,
                        'reasoning': f"Python scan fix for {vuln_id} with constraint {version_constraint}"
                    })
                    self.logger.info(f"📝 Will attempt {container_type} Dockerfile fix: {package} {version_constraint} via pip for {vuln_id} (pyscan)")
        # Apply immediate allowlist fixes by container type
        for container_type, fixes in container_fixes.items():
            if fixes['os_allowlist'] or fixes['py_allowlist']:
                self.logger.info(f"📝 Applying {len(fixes['os_allowlist'])} OS and {len(fixes['py_allowlist'])} Python immediate allowlist fixes to {container_type}...")
                
                if container_type != 'unknown': 
                    allowlist_success = self.apply_allowlist_fixes(container_type, fixes['os_allowlist'], fixes['py_allowlist'])
                    if not allowlist_success:
                        self.logger.error(f"❌ Failed to apply immediate allowlist fixes to {container_type}")
        # Store attempted fixes by container for tracking
        self.attempted_dockerfile_fixes_by_container = container_fixes
        # Apply Dockerfile fixes by container type
        overall_success = True
        total_dockerfile_fixes = 0
        for container_type, fixes in container_fixes.items():
            dockerfile_fixes = fixes['dockerfile_fixes']
            if not dockerfile_fixes:
                self.logger.info(f"ℹ️ No Dockerfile fixes for {container_type}")
                continue
            if container_type == 'unknown':
                self.logger.warning(f"⚠️ Skipping unknown container type fixes - cannot determine target")
                continue
            total_dockerfile_fixes += len(dockerfile_fixes)
            self.logger.info(f"🔧 Applying {len(dockerfile_fixes)} Dockerfile fixes to {container_type}")
            success, vulnerability_mapping = self.apply_dockerfile_fixes(container_type, dockerfile_fixes)
            if not success:
                self.logger.error(f"❌ Failed to apply Dockerfile fixes to {container_type}")
                overall_success = False
            else:
                self.logger.info(f"✅ Applied {len(dockerfile_fixes)} Dockerfile fixes to {container_type}")
        if total_dockerfile_fixes == 0:
            self.logger.info("ℹ️ No Dockerfile fixes to apply - all vulnerabilities pre-filtered to allowlist")
            return True
        self.logger.info(f"📊 Total Dockerfile fixes applied: {total_dockerfile_fixes}")
        return overall_success
    
    def handle_remaining_vulnerabilities_with_container_specific_allowlist(self, pr_number: int, vulnerabilities_by_container: Dict, remaining_failures: List[Dict]) -> bool:
        """Handle vulnerabilities that persist after Dockerfile fixes by reverting and allowlisting"""
        self.logger.info("🔄 Handling remaining vulnerabilities with container-specific reverting and allowlisting...")
        # Get logs for remaining failures to see which vulnerabilities persist
        remaining_logs = self.collect_security_logs(remaining_failures)
        remaining_vulns_by_container = self.extract_all_vulnerabilities_by_container(remaining_logs)
        # Determine which vulnerabilities still exist by container type
        container_persistent_vulns = {
            'training': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'inference': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'unknown': {'os_vulnerabilities': [], 'py_vulnerabilities': []}
        }
        packages_to_revert_by_container = {
            'training': [],
            'inference': [],
            'unknown': []
        }
        # Check each container type for persistent vulnerabilities
        for container_type in ['training', 'inference', 'unknown']:
            original_container_vulns = vulnerabilities_by_container.get(container_type, {'os_vulnerabilities': [], 'py_vulnerabilities': []})
            remaining_container_vulns = remaining_vulns_by_container.get(container_type, {'os_vulnerabilities': [], 'py_vulnerabilities': []})
            # Check OS vulnerabilities
            for vuln in original_container_vulns['os_vulnerabilities']:
                was_dockerfile_attempt = any(
                    fix['vulnerability_id'] == vuln['vulnerability_id'] and fix.get('container_type') == container_type
                    for fix in getattr(self, 'attempted_dockerfile_fixes_by_container', {}).get(container_type, {}).get('dockerfile_fixes', [])
                )
                if was_dockerfile_attempt:
                    still_exists = any(
                        rv['vulnerability_id'] == vuln['vulnerability_id'] and rv['package'] == vuln['package']
                        for rv in remaining_container_vulns['os_vulnerabilities']
                    )
                    if still_exists:
                        container_persistent_vulns[container_type]['os_vulnerabilities'].append(vuln)
                        packages_to_revert_by_container[container_type].append(vuln['package'])
                        self.logger.info(f"🔄 {container_type} OS vulnerability persists: {vuln['vulnerability_id']} in {vuln['package']}")
            # Check Python vulnerabilities
            for vuln in original_container_vulns['py_vulnerabilities']:
                was_dockerfile_attempt = any(
                    fix['vulnerability_id'] == vuln['vulnerability_id'] and fix.get('container_type') == container_type
                    for fix in getattr(self, 'attempted_dockerfile_fixes_by_container', {}).get(container_type, {}).get('dockerfile_fixes', [])
                )
                if was_dockerfile_attempt:
                    still_exists = any(
                        rv['vulnerability_id'] == vuln['vulnerability_id'] and rv['package'] == vuln['package']  
                        for rv in remaining_container_vulns['py_vulnerabilities']
                    )
                    if still_exists:
                        container_persistent_vulns[container_type]['py_vulnerabilities'].append(vuln)
                        packages_to_revert_by_container[container_type].append(vuln['package'])
                        self.logger.info(f"🔄 {container_type} Python vulnerability persists: {vuln['vulnerability_id']} in {vuln['package']}")
        total_persistent = sum(len(container_vulns['os_vulnerabilities']) + len(container_vulns['py_vulnerabilities']) 
                            for container_vulns in container_persistent_vulns.values())
        if total_persistent == 0:
            self.logger.info("✅ No persistent vulnerabilities detected - Dockerfile fixes worked for attempted packages")
            return True
        for container_type, packages_to_revert in packages_to_revert_by_container.items():
            if packages_to_revert and container_type != 'unknown':
                self.logger.info(f"🔄 Reverting {container_type} Dockerfile fixes for {len(packages_to_revert)} packages that didn't work")
                revert_success = self.selectively_revert_dockerfile_packages(container_type, packages_to_revert)
                if not revert_success:
                    self.logger.error(f"❌ Failed to revert packages for {container_type}")
        allowlist_success = True
        for container_type, persistent_vulns in container_persistent_vulns.items():
            if container_type == 'unknown':
                continue
            os_vulns = persistent_vulns['os_vulnerabilities']
            py_vulns = persistent_vulns['py_vulnerabilities']
            if not os_vulns and not py_vulns:
                continue
            os_allowlist_fixes = []
            for vuln in os_vulns:
                os_allowlist_fixes.append({
                    'vulnerability_id': vuln['vulnerability_id'],
                    'package': vuln['package'],
                    'description': vuln['description'],
                    'original_vulnerability_data': vuln.get('original_data')
                })
            py_allowlist_fixes = []
            for vuln in py_vulns:
                py_allowlist_fixes.append({
                    'vulnerability_id': vuln['vulnerability_id'],
                    'description': vuln['description']
                })
            self.logger.info(f"📝 Applying {len(os_allowlist_fixes)} OS and {len(py_allowlist_fixes)} Python allowlist fixes to {container_type}")
            container_success = self.apply_allowlist_fixes(container_type, os_allowlist_fixes, py_allowlist_fixes)
            if not container_success:
                allowlist_success = False
        if allowlist_success:
            commit_msg = f"AutoGluon {self.current_version}: Revert failed container-specific fixes and apply additional allowlists"
            if self.commit_and_push_changes(commit_msg):
                self.logger.info("✅ Reverted failed fixes and applied additional container-specific allowlists")
                if self.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=20):
                    final_failures = self.get_failing_security_tests(pr_number)
                    if not final_failures:
                        self.logger.info("🎉 All vulnerabilities resolved with container-specific approach!")
                        return True
                    else:
                        self.logger.warning(f"⚠️ {len(final_failures)} tests still failing after additional container-specific allowlists")
                        return False
                else:
                    self.logger.warning("⚠️ Timeout waiting for final test results")
                    return False
            else:
                self.logger.error("❌ Failed to commit allowlist changes")
                return False
        else:
            self.logger.error("❌ Failed to apply container-specific allowlist fixes")
            return False
    def _is_autogluon_test(self, test_name: str) -> bool:
        """Detect AutoGluon tests (excluding security tests)"""
        test_name_lower = test_name.lower()
        
        # Skip security tests - we handle those separately
        if self._is_security_test(test_name):
            return False
        
        # AutoGluon test indicators
        autogluon_indicators = [
            'autogluon',
            'ag-',
            'test_autogluon',
            'autogluon-training',
            'autogluon-inference'
        ]
        
        for indicator in autogluon_indicators:
            if indicator in test_name_lower:
                return True
        
        return False

    def get_failing_autogluon_tests(self, pr_number: int) -> List[Dict]:
        """Get failing AutoGluon tests (excluding security tests)"""
        try:
            all_tests = self.get_all_tests_for_pr(pr_number)
            
            failing_autogluon_tests = []
            for test in all_tests:
                # Check if test is AutoGluon-related and failing
                is_autogluon = self._is_autogluon_test(test['name'])
                is_failing = (
                    test.get('status') == 'failure' or 
                    test.get('conclusion') == 'failure' or
                    test.get('state') in ['FAILURE', 'ERROR', 'failure', 'error']
                )
                
                if is_autogluon and is_failing:
                    failing_autogluon_tests.append({
                        'name': test['name'],
                        'check_run_id': test.get('check_run_id'),
                        'url': test.get('url', ''),
                        'details_url': test.get('details_url', ''),
                        'source': test.get('source', 'unknown')
                    })
            
            self.logger.info(f"🔍 Found {len(failing_autogluon_tests)} failing AutoGluon tests")
            for test in failing_autogluon_tests:
                self.logger.info(f"   - {test['name']} (source: {test.get('source', 'unknown')})")
            
            return failing_autogluon_tests
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get failing AutoGluon tests: {e}")
            return []

    def get_all_tests_for_pr(self, pr_number: int) -> List[Dict]:
        """Get all tests for PR (similar to get_all_security_tests but for all tests)"""
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
            
            all_tests = []
            
            # Get tests via GraphQL
            graphql_tests = self._get_graphql_tests_for_commit(pr_number, head_sha)
            for test in graphql_tests:
                # Map GraphQL status to our format
                if test['state'] == 'SUCCESS':
                    mapped_status = 'success'
                elif test['state'] in ['FAILURE', 'ERROR', 'CANCELLED', 'TIMED_OUT']:
                    mapped_status = 'failure'
                elif test['state'] == 'PENDING':
                    mapped_status = 'pending'
                else:
                    mapped_status = test['state'].lower()
                
                all_tests.append({
                    'name': test['name'],
                    'check_run_id': test.get('check_run_id'),
                    'status': mapped_status,
                    'state': test['state'],
                    'url': test.get('url', ''),
                    'details_url': test.get('details_url', ''),
                    'source': 'GraphQL'
                })
            
            return all_tests
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get all tests: {e}")
            return []

    def setup_failure_analysis_chain(self):
        """Setup AI chain for analyzing test failure logs"""
        self.failure_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a test failure analysis AI. Your job is to analyze test failure logs and extract the root cause of failures.

    TASK: Analyze the test failure logs and identify what's causing the test to fail.

    INSTRUCTIONS:
    1. Look for the "Failures" section in the logs
    2. Extract the most relevant error messages, stack traces, and failure reasons
    3. Identify the root cause (e.g., import errors, assertion failures, timeout issues, dependency conflicts)
    4. Provide a concise summary of what went wrong
    5. If possible, suggest what type of fix might be needed

    FOCUS ON:
    - Error messages and exceptions
    - Failed assertions
    - Import/dependency issues
    - Timeout or resource issues
    - Configuration problems
    - Environment setup issues

    OUTPUT FORMAT:
    Return a JSON object with:
    {{
        "root_cause": "Brief description of the main issue",
        "error_type": "import_error|assertion_failure|timeout|dependency_conflict|config_issue|other",
        "key_errors": ["list", "of", "key", "error", "messages"],
        "suggested_fix_type": "Brief suggestion for fix type",
        "failure_section": "Extracted relevant failure content"
    }}

    Only analyze the failure content, don't try to fix the issues."""),
            ("human", """Test Name: {test_name}

    Test Failure Logs:
    {failure_logs}

    Analyze the failure and extract the root cause:""")
        ])
        
        self.failure_analysis_chain = self.failure_analysis_prompt | self.llm | JsonOutputParser()

    def extract_failure_logs_from_test(self, test_name: str, test_url: str) -> str:
        """Extract failure logs from test URL, focusing on Failures section"""
        try:
            self.logger.info(f"🔍 Extracting failure logs for {test_name}")
            logs = self.get_logs_from_test_url(test_url, test_name)
            if not logs:
                return ""
            failure_extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract the Failures section from test logs. Look for content between "Failures" and "Passes" sections, or similar failure indicators. Return only the relevant failure content without the surrounding log noise."""),
                ("human", """Test logs:
    {logs}

    Extract the failures section:""")
            ])
            
            try:
                failure_chain = failure_extraction_prompt | self.llm
                failure_content = failure_chain.invoke({"logs": logs[:50000]})
                
                if failure_content and len(str(failure_content).strip()) > 10:
                    self.logger.info(f"✅ Extracted failure content for {test_name} ({len(str(failure_content))} chars)")
                    return str(failure_content)
                else:
                    self.logger.warning(f"⚠️ AI extraction returned minimal content for {test_name}")
                    return logs[:10000] 
                    
            except Exception as e:
                self.logger.warning(f"⚠️ AI failure extraction failed for {test_name}: {e}")
                return logs[:10000] 
                
        except Exception as e:
            self.logger.error(f"❌ Failed to extract failure logs for {test_name}: {e}")
            return ""

    def analyze_autogluon_test_failures(self, failing_tests: List[Dict]) -> Dict:
        """Analyze AutoGluon test failures using AI"""
        if not hasattr(self, 'failure_analysis_chain'):
            self.setup_failure_analysis_chain()
        
        self.logger.info(f"🤖 Analyzing {len(failing_tests)} AutoGluon test failures...")
        
        analysis_results = {}
        
        for test in failing_tests:
            test_name = test['name']
            self.logger.info(f"🔍 Analyzing failure: {test_name}")
            
            # Extract failure logs
            failure_logs = ""
            if test.get('url'):
                failure_logs = self.extract_failure_logs_from_test(test_name, test['url'])
            if not failure_logs and test.get('details_url'):
                failure_logs = self.extract_failure_logs_from_test(test_name, test['details_url'])
            
            if not failure_logs:
                self.logger.warning(f"⚠️ No failure logs found for {test_name}")
                continue
            
            try:
                # Use AI to analyze the failure
                analysis = self.failure_analysis_chain.invoke({
                    "test_name": test_name,
                    "failure_logs": failure_logs[:30000]  # Limit for AI processing
                })
                
                if isinstance(analysis, dict):
                    analysis_results[test_name] = analysis
                    self.logger.info(f"✅ Analyzed {test_name}: {analysis.get('root_cause', 'Unknown cause')}")
                else:
                    self.logger.warning(f"⚠️ AI returned invalid analysis for {test_name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to analyze {test_name}: {e}")
                continue
        if analysis_results:
            self.logger.info("="*80)
            self.logger.info("🔍 AUTOGLUON TEST FAILURE ANALYSIS SUMMARY")
            self.logger.info("="*80)
            
            error_types = {}
            for test_name, analysis in analysis_results.items():
                error_type = analysis.get('error_type', 'unknown')
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(test_name)
                
                self.logger.info(f"📋 {test_name}:")
                self.logger.info(f"   Root Cause: {analysis.get('root_cause', 'Unknown')}")
                self.logger.info(f"   Error Type: {analysis.get('error_type', 'unknown')}")
                self.logger.info(f"   Suggested Fix: {analysis.get('suggested_fix_type', 'Unknown')}")
                if analysis.get('key_errors'):
                    self.logger.info(f"   Key Errors: {', '.join(analysis['key_errors'][:3])}")
            
            self.logger.info("\n📊 Error Type Summary:")
            for error_type, tests in error_types.items():
                self.logger.info(f"   {error_type}: {len(tests)} tests")
            
            self.logger.info("="*80)
        
        return analysis_results

    def check_autogluon_test_failures(self, pr_number: int) -> bool:
        """Check for AutoGluon test failures and analyze them"""
        self.logger.info("🔍 Checking for AutoGluon test failures...")
        
        failing_autogluon_tests = self.get_failing_autogluon_tests(pr_number)
        
        if not failing_autogluon_tests:
            self.logger.info("✅ No failing AutoGluon tests found!")
            return True
        
        # Analyze the failures
        failure_analysis = self.analyze_autogluon_test_failures(failing_autogluon_tests)
        
        if failure_analysis:
            self.logger.warning(f"⚠️ Found {len(failing_autogluon_tests)} failing AutoGluon tests")
            self.logger.info("ℹ️ AutoGluon test failures detected but not blocking security analysis")
            return True 
        return True
    
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
def main():
    """Main function for Security Test Agent"""
    import argparse
    parser = argparse.ArgumentParser(description='AutoGluon Security Test Agent')
    parser.add_argument('--current-version', required=True, help='Current version (e.g., 1.3.1)')
    parser.add_argument('--previous-version', required=True, help='Previous version (e.g., 1.3.0)')
    parser.add_argument('--fork-url', required=True, help='Your fork URL')
    parser.add_argument('--debug-tests', action='store_true', help='Run debug analysis of available tests only')
    parser.add_argument('--pr-number', type=int, help='Specific PR number to debug (optional)')
    args = parser.parse_args()
    agent = SecurityTestAgent(
        args.current_version,
        args.previous_version,
        args.fork_url
    )
    success = agent.run_security_test_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()