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

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from common import BaseAutomation

@dataclass
class SecurityIssue:
    package: str
    vulnerability_id: str
    description: str
    severity: str
    test_type: str  # 'os_scan', 'py_scan', or 'general_security'
    fix_action: str

class SecurityFixPlan(BaseModel):
    """Enhanced model for agentic security analysis responses"""
    os_scan_issues: List[Dict] = Field(description="OS scan vulnerability issues with AI analysis")
    py_scan_issues: List[Dict] = Field(description="Python scan vulnerability issues with AI analysis")
    dockerfile_fixes: List[Dict] = Field(description="AI-recommended Dockerfile fixes with reasoning")
    os_scan_allowlist_fixes: List[Dict] = Field(description="AI-recommended OS scan allowlist fixes")
    py_scan_allowlist_fixes: List[Dict] = Field(description="AI-recommended Python scan allowlist fixes")
    try_dockerfile_first: bool = Field(description="AI decision: try Dockerfile fixes before allowlist")
    severity_assessment: str = Field(description="AI's overall security risk assessment")
    confidence_level: str = Field(description="AI confidence level: HIGH/MEDIUM/LOW", default="MEDIUM")
    alternative_approaches: str = Field(description="Other strategies AI considered", default="")
    
    class Config:
        extra = "allow"  

class SecurityTestAgent(BaseAutomation):
    """Agentic system for automatically fixing security test failures using selective reversion approach"""
    
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
        
    def setup_github_credentials(self):
        """Setup GitHub credentials for API access"""
        self.github_token = os.environ.get('GITHUB_TOKEN')
        if not self.github_token:
            try:
                result = subprocess.run(
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
        
        # Use specific environment variables for Bedrock if available
        bedrock_access_key = os.getenv('BEDROCK_AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID')
        bedrock_secret_key = os.getenv('BEDROCK_AWS_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
        bedrock_region = os.getenv('BEDROCK_REGION') or os.getenv('REGION', 'us-east-1')
        bedrock_session_token = os.getenv('BEDROCK_AWS_SESSION_TOKEN') or os.getenv('AWS_SESSION_TOKEN')
        
        try:
            # Create Bedrock client with specific credentials if provided
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
        
        # Use specific environment variables for CodeBuild if available
        codebuild_access_key = os.getenv('CODEBUILD_AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID')
        codebuild_secret_key = os.getenv('CODEBUILD_AWS_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')
        codebuild_region = os.getenv('CODEBUILD_REGION') or os.getenv('REGION', 'us-west-2')
        codebuild_session_token = os.getenv('CODEBUILD_AWS_SESSION_TOKEN') or os.getenv('AWS_SESSION_TOKEN')
        
        try:
            # Create CodeBuild client with specific credentials if provided
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
        
    # REPLACE the setup_langchain method with this FIXED version:

    def setup_langchain(self):
        """Initialize LangChain with Claude via Bedrock - AI FOR DETECTION ONLY"""
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
        
        # SIMPLIFIED DETECTION-ONLY PROMPT
        self.detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a vulnerability detection AI. Your ONLY job is to extract vulnerability information from security scan logs.

    ## WHAT TO EXTRACT

    ### OS Scan Vulnerabilities (ECR Enhanced Scan):
    Look for JSON patterns with vulnerability_id and package names:
    Total of X vulnerabilities need to be fixed on [container]:
    {{"package_name": [{{"vulnerability_id": "CVE-YYYY-NNNN", "description": "...", "severity": "..."}}]}}

    ### Python Scan Vulnerabilities (Safety Reports):
    Look for SAFETY_REPORT patterns:
    SAFETY_REPORT (FAILED) [pkg: package_name] [...] vulnerability_id='12345' [...] advisory='...'

    ## OUTPUT FORMAT

    Return ONLY a simple JSON list of detected vulnerabilities:

    ```json
    {{
    "os_vulns": [
        {{
        "package": "package_name",
        "vulnerability_id": "CVE-YYYY-NNNN",
        "description": "description text",
        "severity": "CRITICAL|HIGH|MEDIUM|LOW"
        }}
    ],
    "py_vulns": [
        {{
        "package": "package_name",
        "vulnerability_id": "vulnerability_id",
        "description": "description text",
        "severity": "UNKNOWN"
        }}
    ]
    }}
    INSTRUCTIONS

    Extract ALL vulnerabilities from both OS scan JSON and Python Safety reports
    Be precise - only extract what you can clearly identify
    Don't make decisions - just detect and extract
    Use exact vulnerability IDs and package names from the logs
    Keep descriptions short - main vulnerability text only

    Focus on accurate detection, not analysis or recommendations."""),
    ("human", """Extract all vulnerabilities from these security logs:
    Security Test Logs:
    {security_logs}
    Extract every vulnerability you can find - both OS scan CVEs and Python Safety reports. Return only the detection results in JSON format.""")
    ])
        self.detection_chain = self.detection_prompt | self.llm | JsonOutputParser()


    def debug_all_available_tests(self, pr_number: int = None) -> None:
        """DEBUG METHOD: Show current test status clearly"""
        self.logger.info("🔍 DEBUG: Current Test Status Analysis")
        self.logger.info("="*80)
        
        if pr_number is None:
            pr_number = self.get_current_pr_number()
            if not pr_number:
                self.logger.error("❌ No PR found for debugging")
                return
        
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
            
            self.logger.info(f"📋 PR #{pr_number}: {pr_data['title']}")
            self.logger.info(f"📋 Current HEAD SHA: {head_sha}")
            self.logger.info(f"📋 State: {pr_data['state']}")
            
            # Get current test status (HEAD commit only)
            current_tests = {}
            
            # 1. Get REST API Check Runs for HEAD only
            self.logger.info(f"\n🔍 REST API Check Runs (HEAD commit only):")
            check_runs_url = f"https://api.github.com/repos/aws/deep-learning-containers/commits/{head_sha}/check-runs"
            response = requests.get(check_runs_url, headers=headers, params={"per_page": 100})
            
            if response.status_code == 200:
                check_runs = response.json().get('check_runs', [])
                self.logger.info(f"   Found {len(check_runs)} check runs")
                
                for check in check_runs:
                    check_name = check['name']
                    is_security = self._is_security_test(check_name)
                    
                    current_tests[check_name] = {
                        'name': check_name,
                        'source': 'REST_check_run',
                        'status': check['status'],
                        'conclusion': check.get('conclusion'),
                        'is_security': is_security,
                        'is_failing': check.get('conclusion') == 'failure'
                    }
                    
                    status_display = f"{check['status']}/{check.get('conclusion', 'N/A')}"
                    security_indicator = "🛡️ SECURITY" if is_security else "   regular"
                    failing_indicator = "❌" if check.get('conclusion') == 'failure' else "✅" if check.get('conclusion') == 'success' else "⏳"
                    self.logger.info(f"   {failing_indicator} {security_indicator}: {check_name} ({status_display})")
            else:
                self.logger.warning(f"   ⚠️ Failed to get check runs: {response.status_code}")
            
            # 2. Get REST API Commit Statuses for HEAD only
            self.logger.info(f"\n🔍 REST API Commit Statuses (HEAD commit only):")
            status_url = f"https://api.github.com/repos/aws/deep-learning-containers/commits/{head_sha}/status"
            response = requests.get(status_url, headers=headers)
            
            if response.status_code == 200:
                status_data = response.json()
                statuses = status_data.get('statuses', [])
                self.logger.info(f"   Found {len(statuses)} commit statuses")
                
                for status in statuses:
                    status_name = status['context']
                    is_security = self._is_security_test(status_name)
                    
                    current_tests[status_name] = {
                        'name': status_name,
                        'source': 'REST_status',
                        'state': status['state'],
                        'is_security': is_security,
                        'is_failing': status['state'] == 'failure'
                    }
                    
                    security_indicator = "🛡️ SECURITY" if is_security else "   regular"
                    failing_indicator = "❌" if status['state'] == 'failure' else "✅" if status['state'] == 'success' else "⏳"
                    self.logger.info(f"   {failing_indicator} {security_indicator}: {status_name} ({status['state']})")
            else:
                self.logger.warning(f"   ⚠️ Failed to get commit statuses: {response.status_code}")
            
            # 2.5. Check GraphQL for current HEAD (sometimes shows tests REST API doesn't)
            self.logger.info(f"\n🔍 GraphQL API (current HEAD only):")
            graphql_tests = self._get_graphql_tests_for_commit(pr_number, head_sha)
            if graphql_tests:
                self.logger.info(f"   Found {len(graphql_tests)} tests via GraphQL for HEAD")
                for test in graphql_tests:
                    test_name = test['name']
                    is_security = self._is_security_test(test_name)
                    
                    # Add to current_tests if not already there
                    if test_name not in current_tests:
                        current_tests[test_name] = {
                            'name': test_name,
                            'source': 'GraphQL',
                            'state': test.get('state', 'UNKNOWN'),
                            'is_security': is_security,
                            'is_failing': test.get('state') in ['FAILURE', 'ERROR'],
                            'check_run_id': test.get('check_run_id')  # Include check_run_id from enhanced GraphQL
                        }
                        
                        security_indicator = "🛡️ SECURITY" if is_security else "   regular"
                        failing_indicator = "❌" if test.get('state') in ['FAILURE', 'ERROR'] else "✅" if test.get('state') == 'SUCCESS' else "⏳"
                        
                        # Show available URLs for debugging
                        url_info = ""
                        if test.get('check_run_id'):
                            url_info += f" (ID: {test.get('check_run_id')})"
                        if test.get('url'):
                            url_info += f" [URL: {test.get('url')[:50]}...]"
                        if test.get('details_url'):
                            url_info += f" [Details: {test.get('details_url')[:50]}...]"
                        if not url_info:
                            url_info = " (no access info)"
                            
                        self.logger.info(f"   {failing_indicator} {security_indicator}: {test_name} ({test.get('state', 'UNKNOWN')}){url_info}")
                    else:
                        self.logger.info(f"   ↳ (already found in REST API): {test_name}")
            else:
                self.logger.info(f"   No additional tests found via GraphQL for HEAD")
            
            # 2.6. Check if AutoGluon tests are completely missing (might not have started)
            autogluon_test_names = [name for name in current_tests.keys() if 'autogluon' in name.lower()]
            if not autogluon_test_names:
                self.logger.info(f"\n⚠️ AUTOGLUON TESTS MISSING:")
                self.logger.info(f"   No AutoGluon tests detected in current HEAD commit")
                self.logger.info(f"   This could mean:")
                self.logger.info(f"   - AutoGluon tests haven't started yet")
                self.logger.info(f"   - Tests are queued but not visible")
                self.logger.info(f"   - PR doesn't trigger AutoGluon tests")
                
                # Check if we can find any AutoGluon tests in recent history
                self.logger.info(f"\n🔍 Checking recent commits for AutoGluon tests...")
                recent_autogluon_tests = self._get_recent_autogluon_tests(pr_number)
                if recent_autogluon_tests:
                    self.logger.info(f"   Found AutoGluon tests in recent commits:")
                    for test in recent_autogluon_tests:
                        security_indicator = "🛡️ SECURITY" if self._is_security_test(test['name']) else "   regular"
                        self.logger.info(f"   {security_indicator}: {test['name']} ({test.get('state', 'UNKNOWN')}) [commit: {test.get('commit_oid', 'unknown')[:8]}]")
                else:
                    self.logger.info(f"   No AutoGluon tests found in recent commits either")
            
            # 3. Analysis Summary
            self.logger.info(f"\n📊 CURRENT TEST SUMMARY:")
            self.logger.info("="*50)
            
            total_tests = len(current_tests)
            security_tests = [t for t in current_tests.values() if t['is_security']]
            failing_tests = [t for t in current_tests.values() if t.get('is_failing', False)]
            failing_security_tests = [t for t in security_tests if t.get('is_failing', False)]
            
            self.logger.info(f"Total current tests: {total_tests}")
            self.logger.info(f"Security tests: {len(security_tests)}")
            self.logger.info(f"Failing tests: {len(failing_tests)}")
            self.logger.info(f"Failing security tests: {len(failing_security_tests)}")
            
            # Show current failing tests
            if failing_tests:
                self.logger.info(f"\n❌ CURRENTLY FAILING TESTS ({len(failing_tests)}):")
                for test in failing_tests:
                    security_indicator = "🛡️ SECURITY" if test['is_security'] else "   regular"
                    self.logger.info(f"   {security_indicator}: {test['name']}")
            else:
                self.logger.info(f"\n✅ No tests currently failing")
            
            # Show current security tests
            if security_tests:
                self.logger.info(f"\n🛡️ CURRENT SECURITY TESTS ({len(security_tests)}):")
                for test in security_tests:
                    status = test.get('status', test.get('state', 'UNKNOWN'))
                    conclusion = test.get('conclusion', '')
                    if conclusion:
                        status_display = f"{status}/{conclusion}"
                    else:
                        status_display = status
                    
                    failing_indicator = "❌" if test.get('is_failing', False) else "✅"
                    self.logger.info(f"   {failing_indicator} {test['name']} ({status_display})")
            else:
                self.logger.info(f"\n⚠️ No security tests detected!")
            
            # Show AutoGluon tests
            autogluon_tests = [t for t in current_tests.values() if 'autogluon' in t['name'].lower()]
            if autogluon_tests:
                self.logger.info(f"\n🎯 AUTOGLUON TESTS ({len(autogluon_tests)}):")
                for test in autogluon_tests:
                    security_indicator = "🛡️ SECURITY" if test['is_security'] else "   regular"
                    failing_indicator = "❌" if test.get('is_failing', False) else "✅"
                    status = test.get('status', test.get('state', 'UNKNOWN'))
                    self.logger.info(f"   {failing_indicator} {security_indicator}: {test['name']} ({status})")
            else:
                self.logger.info(f"\n⚠️ NO AUTOGLUON TESTS FOUND ON CURRENT HEAD")
                self.logger.info(f"   This means your security test agent won't find anything to process")
                self.logger.info(f"   Possible reasons:")
                self.logger.info(f"   1. AutoGluon tests are still queued/starting")
                self.logger.info(f"   2. Tests failed to trigger properly")
                self.logger.info(f"   3. Need to wait longer for tests to appear")
                self.logger.info(f"   4. PR configuration issue")
            
            self.logger.info("="*80)
            self.logger.info("🔍 DEBUG ANALYSIS COMPLETE")
            
            # Show what your security test agent would find
            self.logger.info(f"\n🤖 WHAT SECURITY AGENT WOULD DETECT:")
            failing_security_for_agent = [t for t in current_tests.values() 
                                        if t['is_security'] and t.get('is_failing', False)]
            if failing_security_for_agent:
                self.logger.info(f"   Would process {len(failing_security_for_agent)} failing security tests:")
                for test in failing_security_for_agent:
                    self.logger.info(f"   - {test['name']}")
            else:
                self.logger.info(f"   ✅ No failing security tests to process")
                if not autogluon_tests:
                    self.logger.info(f"   ⚠️ RECOMMENDATION: Wait for AutoGluon tests to start before running security agent")
            
        except Exception as e:
            self.logger.error(f"❌ Debug analysis failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")

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
                # Process check suites
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
                        
                        # Try to get check run ID for log access
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
                        
                        # Log if we found an ID
                        if check_run_id:
                            self.logger.info(f"🔍 GraphQL found check_run_id for {run['name']}: {check_run_id}")
                
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

    def _get_recent_autogluon_tests(self, pr_number: int) -> List[Dict]:
        """Look for AutoGluon tests in recent commits of the PR"""
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              commits(last: 5) {
                nodes {
                  commit {
                    oid
                    checkSuites(first: 50) {
                      nodes {
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
            "owner": "aws",
            "repo": "deep-learning-containers",
            "number": pr_number
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
                return []
            
            autogluon_tests = []
            commits = data['data']['repository']['pullRequest']['commits']['nodes']
            
            for commit_node in commits:
                commit = commit_node['commit']
                oid = commit['oid']
                
                # Check check suites
                check_suites = commit.get('checkSuites', {}).get('nodes', [])
                for suite in check_suites:
                    check_runs = suite.get('checkRuns', {}).get('nodes', [])
                    for run in check_runs:
                        if 'autogluon' in run['name'].lower():
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
                            
                            autogluon_tests.append({
                                'name': run['name'],
                                'state': state,
                                'commit_oid': oid,
                                'type': 'check_run'
                            })
                
                # Check commit statuses
                status = commit.get('status')
                if status:
                    contexts = status.get('contexts', [])
                    for context in contexts:
                        if 'autogluon' in context['context'].lower():
                            autogluon_tests.append({
                                'name': context['context'],
                                'state': context['state'].upper(),
                                'commit_oid': oid,
                                'type': 'status_context'
                            })
            
            return autogluon_tests
            
        except Exception as e:
            self.logger.warning(f"Failed to get recent AutoGluon tests: {e}")
            return []

    def _is_security_test(self, test_name: str) -> bool:
        """Enhanced security test detection with detailed logging"""
        test_name_lower = test_name.lower()
        
        # Skip dlc-pr-quick-checks
        if 'dlc-pr-quick-checks' in test_name_lower:
            return False
        
        # Check security criteria
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

    def _debug_get_tests_via_graphql(self, pr_number: int) -> List[Dict]:
        """Debug version of GraphQL test retrieval with enhanced logging"""
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
            "owner": "aws",
            "repo": "deep-learning-containers",
            "number": pr_number
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
                self.logger.error(f"❌ GraphQL errors: {data['errors']}")
                return []
            
            tests = []
            pr_data = data['data']['repository']['pullRequest']
            commits = pr_data['commits']['nodes']
            
            self.logger.info(f"📊 GraphQL found {len(commits)} commits to analyze")
            
            for commit_node in commits:
                commit = commit_node['commit']
                oid = commit['oid']
                
                # Process check suites
                check_suites = commit.get('checkSuites', {}).get('nodes', [])
                for suite in check_suites:
                    app_name = suite.get('app', {}).get('name', 'Unknown')
                    check_runs = suite.get('checkRuns', {}).get('nodes', [])
                    
                    for run in check_runs:
                        conclusion = run.get('conclusion', '').upper()
                        status = run.get('status', '').upper()
                        
                        # Determine state
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
                
                # Process commit statuses
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

    def is_package_in_allowlist(self, package_name: str, container_type: str, device_type: str) -> Dict[str, bool]:
        """Check if package exists in either OS or Python scan allowlists"""
        try:
            os_allowlist = self.get_allowlist_content(container_type, device_type, 'os_scan')
            py_allowlist = self.get_allowlist_content(container_type, device_type, 'py_scan')
            
            # Check OS allowlist (packages are keys in the JSON)
            in_os_allowlist = package_name.lower() in os_allowlist
            
            # Check Python allowlist (scan for package name in vulnerability descriptions/IDs)
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
        
        # Check if already in allowlists for any container type
        for container_type in ['training', 'inference']:
            for device_type in ['cpu', 'gpu']:
                allowlist_status = self.is_package_in_allowlist(package_name, container_type, device_type)
                self.logger.info(f"jupyter_core in {container_type}/{device_type} allowlist: {allowlist_status}")
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
        """Get failing security test information from PR - COMPREHENSIVE VERSION"""
        try:
            # Use the comprehensive method to get all security tests
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
                    # Ensure we have a check_run_id for log retrieval if available
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

    def get_test_logs(self, check_run_id: str) -> str:
        """Get logs from a specific check run (handles both GitHub Actions and CodeBuild)"""
        try:
            # Handle both string and int check_run_id from GraphQL
            if isinstance(check_run_id, str) and not check_run_id.isdigit():
                # This might be a GraphQL node ID, try to extract numeric ID or use as-is
                self.logger.info(f"🔍 Working with GraphQL check run ID: {check_run_id}")
                
                # For GraphQL node IDs, we might need to get the numeric run ID differently
                # Try to use the check run details API with the node ID
                check_run_url = f"https://api.github.com/repos/aws/deep-learning-containers/actions/runs/{check_run_id}"
                headers = {
                    "Authorization": f"Bearer {self.github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
                
                # Try to get check run details first
                response = requests.get(check_run_url, headers=headers)
                if response.status_code == 200:
                    check_run_data = response.json()
                    # Look for logs_url or try logs endpoint
                    logs_url = f"https://api.github.com/repos/aws/deep-learning-containers/actions/runs/{check_run_id}/logs"
                    logs_response = requests.get(logs_url, headers=headers, allow_redirects=True)
                    
                    if logs_response.status_code == 200:
                        self.logger.info(f"✅ Retrieved GitHub Actions logs for check run {check_run_id}")
                        return logs_response.text
                
                # If GitHub Actions doesn't work, try CodeBuild approach
                self.logger.info(f"GitHub Actions logs not available, trying CodeBuild for {check_run_id}")
                return self.get_codebuild_logs_for_check_run(check_run_id)
            else:
                # Convert to string for API calls
                check_run_id_str = str(check_run_id)
                
                # First try to get GitHub Actions logs
                url = f"https://api.github.com/repos/aws/deep-learning-containers/actions/runs/{check_run_id_str}/logs"
                headers = {
                    "Authorization": f"Bearer {self.github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
                
                response = requests.get(url, headers=headers, allow_redirects=True)
                
                if response.status_code == 200:
                    self.logger.info(f"✅ Retrieved GitHub Actions logs for check run {check_run_id}")
                    return response.text
                else:
                    # Try to get CodeBuild logs if GitHub Actions logs not available
                    self.logger.info(f"GitHub Actions logs not available, trying CodeBuild for check run {check_run_id}")
                    return self.get_codebuild_logs_for_check_run(check_run_id_str)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get logs for check run {check_run_id}: {e}")
            return ""

    def get_codebuild_logs_for_check_run(self, check_run_id: str) -> str:
        """Get CodeBuild logs for a specific check run"""
        try:
            # Get check run details to find CodeBuild URL
            check_run_url = f"https://api.github.com/repos/aws/deep-learning-containers/actions/runs/{check_run_id}"
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            
            response = requests.get(check_run_url, headers=headers)
            if response.status_code != 200:
                self.logger.warning(f"⚠️ Could not get check run details for {check_run_id}")
                return ""
            
            check_run_data = response.json()
            
            # Look for CodeBuild URL in the check run details
            codebuild_url = None
            
            # Check in details_url
            if 'details_url' in check_run_data and 'codebuild' in check_run_data['details_url']:
                codebuild_url = check_run_data['details_url']
            
            # Check in logs_url if available
            if not codebuild_url and 'logs_url' in check_run_data and 'codebuild' in check_run_data['logs_url']:
                codebuild_url = check_run_data['logs_url']
            
            if not codebuild_url:
                self.logger.warning(f"⚠️ No CodeBuild URL found for check run {check_run_id}")
                return ""
            
            return self.get_codebuild_logs_from_url(codebuild_url)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get CodeBuild logs for check run {check_run_id}: {e}")
            return ""

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
            # Extract build ID from URL - handle the complex CodeBuild URL format
            # URL format: .../build/PROJECT_NAME%3ABUILD_ID/log?...
            # Where %3A is URL-encoded ':'
            import urllib.parse
            
            build_id = None
            
            # Fix the regex patterns to correctly extract the build ID
            patterns = [
                # More specific pattern for CodeBuild URLs
                r'/codesuite/codebuild/projects/[^/]+/build/([^/?]+)',  # Match everything after /build/ until / or ?
                r'/build/([^/?]+)(?:/log)?',  # Everything between /build/ and next / or ? (with optional /log)
                r'build/([^/?&]+)',  # Everything after build/ until /, ?, or &
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    raw_build_id = match.group(1)
                    # URL decode the build ID
                    build_id = urllib.parse.unquote(raw_build_id)
                    self.logger.info(f"📋 Extracted build ID: {build_id} (from pattern: {pattern})")
                    break
            
            if build_id:
                # Try to get logs via CodeBuild API
                try:
                    self.logger.info(f"🔍 Attempting CodeBuild API with build ID: {build_id}")
                    response = self.codebuild_client.batch_get_builds(ids=[build_id])
                    if response['builds']:
                        build = response['builds'][0]
                        if 'logs' in build and 'groupName' in build['logs']:
                            log_group = build['logs']['groupName']
                            log_stream = build['logs']['streamName']
                            
                            self.logger.info(f"📥 Getting logs from CloudWatch: {log_group}/{log_stream}")
                            log_response = self.logs_client.get_log_events(
                                logGroupName=log_group,
                                logStreamName=log_stream
                            )
                            
                            log_lines = [event['message'] for event in log_response['events']]
                            logs_content = '\n'.join(log_lines)
                            self.logger.info(f"✅ Retrieved {len(log_lines)} log lines from CodeBuild API")
                            return logs_content
                        else:
                            self.logger.warning("⚠️ Build found but no CloudWatch logs configured")
                    else:
                        self.logger.warning(f"⚠️ No build found with ID: {build_id}")
                except Exception as e:
                    self.logger.warning(f"⚠️ CodeBuild API access failed: {e}")
            else:
                self.logger.warning("⚠️ Could not extract build ID from URL")
            
            # Fallback: try to extract logs from page HTML content
            self.logger.info("🔍 Falling back to HTML parsing of CodeBuild page...")
            
            # Look for various log content patterns in the HTML
            log_patterns = [
                # CodeBuild console specific patterns
                r'"logEvents":\s*\[(.*?)\]',
                r'<pre[^>]*class="[^"]*log[^"]*"[^>]*>(.*?)</pre>',
                r'<div[^>]*class="[^"]*log[^"]*"[^>]*>(.*?)</div>',
                r'<code[^>]*>(.*?)</code>',
                r'<pre[^>]*>(.*?)</pre>',
                # Look for error/security content specifically
                r'(CVE-\d{4}-\d{4,}.*?)(?=CVE-\d{4}-\d{4,}|\n\n|$)',
                r'(ERROR.*?)(?=ERROR|\n\n|$)',
                r'(FAILED.*?)(?=FAILED|\n\n|$)',
                r'(vulnerability_id.*?)(?=vulnerability_id|\n\n|$)',
            ]
            
            extracted_content = []
            
            for pattern in log_patterns:
                matches = re.findall(pattern, page_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    self.logger.info(f"📋 Found {len(matches)} matches with pattern: {pattern[:30]}...")
                    for match in matches:
                        # Clean up HTML tags and entities
                        if isinstance(match, tuple):
                            match = match[0] if match else ""
                        
                        clean_content = re.sub(r'<[^>]+>', '', str(match))
                        clean_content = clean_content.replace('&lt;', '<').replace('&gt;', '>')
                        clean_content = clean_content.replace('&amp;', '&').replace('&quot;', '"')
                        clean_content = clean_content.replace('\\n', '\n').replace('\\t', '\t')
                        clean_content = clean_content.strip()
                        
                        if len(clean_content) > 50:  # Substantial content
                            extracted_content.append(clean_content)
            
            if extracted_content:
                combined_logs = '\n---\n'.join(extracted_content)
                self.logger.info(f"✅ Extracted logs from CodeBuild HTML ({len(combined_logs)} chars)")
                return combined_logs
            else:
                # Last resort: return any text content that might contain security info
                text_content = re.sub(r'<[^>]+>', '', page_content)
                if any(keyword in text_content.lower() for keyword in ['cve', 'vulnerability', 'security', 'error', 'failed']):
                    self.logger.info(f"✅ Found security-related text content ({len(text_content)} chars)")
                    return text_content
                else:
                    self.logger.warning("⚠️ No log content found in CodeBuild page")
                    return ""
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract CodeBuild logs: {e}")
            return ""

    def extract_logs_from_github_actions_page(self, page_content: str, url: str) -> str:
        """Extract logs from GitHub Actions page"""
        try:
            # Look for download links or log content in the page
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
                # Clean up any HTML entities
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
            # Look for common log content patterns
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

    def get_codebuild_logs_from_url(self, codebuild_url: str) -> str:
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
            
            # Fallback: try to extract logs from page HTML
            log_patterns = [
                r'<pre[^>]*>(.*?)</pre>',
                r'<code[^>]*>(.*?)</code>',
                r'"logEvents":\s*\[(.*?)\]',
                r'log-content[^>]*>(.*?)</div>'
            ]
            
            for pattern in log_patterns:
                matches = re.findall(pattern, page_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Clean up HTML and return first substantial match
                    for match in matches:
                        clean_content = re.sub(r'<[^>]+>', '', match)
                        clean_content = clean_content.replace('&lt;', '<').replace('&gt;', '>')
                        clean_content = clean_content.replace('&amp;', '&').replace('&quot;', '"')
                        if len(clean_content) > 100:  # Substantial content
                            self.logger.info(f"✅ Extracted logs from CodeBuild page ({len(clean_content)} chars)")
                            return clean_content
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract CodeBuild logs: {e}")
            return ""

    def extract_logs_from_github_actions_page(self, page_content: str, url: str) -> str:
        """Extract logs from GitHub Actions page"""
        try:
            # Look for download links or log content in the page
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
                # Clean up any HTML entities
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
            # Look for common log content patterns
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
        """Extract logs from CodeBuild URL"""
        try:
            self.logger.info(f"🔍 Extracting CodeBuild logs from: {codebuild_url}")
            
            # Parse CodeBuild URL to extract project name and build ID
            # URL format: https://us-west-2.console.aws.amazon.com/codesuite/codebuild/projects/{project}/build/{build_id}
            import re
            
            # Extract project name and build ID from URL
            url_pattern = r'projects/([^/]+)/build/([^/?]+)'
            match = re.search(url_pattern, codebuild_url)
            
            if not match:
                self.logger.warning(f"⚠️ Could not parse CodeBuild URL: {codebuild_url}")
                return ""
            
            project_name = match.group(1)
            build_id = match.group(2)
            
            self.logger.info(f"📋 Found CodeBuild project: {project_name}, build: {build_id}")
            
            # Get build details from CodeBuild
            try:
                response = self.codebuild_client.batch_get_builds(ids=[build_id])
                
                if not response['builds']:
                    self.logger.warning(f"⚠️ No build found with ID: {build_id}")
                    return ""
                
                build = response['builds'][0]
                
                # Get logs from CloudWatch if available
                if 'logs' in build and 'groupName' in build['logs']:
                    log_group = build['logs']['groupName']
                    log_stream = build['logs']['streamName']
                    
                    self.logger.info(f"📥 Getting logs from CloudWatch: {log_group}/{log_stream}")
                    
                    log_response = self.logs_client.get_log_events(
                        logGroupName=log_group,
                        logStreamName=log_stream
                    )
                    
                    # Combine all log events
                    log_lines = []
                    for event in log_response['events']:
                        log_lines.append(event['message'])
                    
                    logs_content = '\n'.join(log_lines)
                    self.logger.info(f"✅ Retrieved {len(log_lines)} log lines from CodeBuild")
                    return logs_content
                else:
                    self.logger.warning(f"⚠️ No CloudWatch logs found for build {build_id}")
                    return ""
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to get CodeBuild details: {e}")
                return ""
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get CodeBuild logs from URL: {e}")
            return ""

    def extract_security_issues_from_logs(self, logs: str, test_name: str) -> List[SecurityIssue]:
        """Extract security vulnerability issues from test logs - FIXED FOR FULL CONTENT"""
        issues = []
        
        self.logger.info(f"📊 Parsing {len(logs)} chars from {test_name}")
        
        if 'test_ecr_enhanced_scan' in test_name or 'enhanced_scan' in test_name.lower():
            # Parse ECR Enhanced Scan issues (from AssertionError JSON format)
            self.logger.info(f"📊 Parsing ECR Enhanced Scan vulnerabilities from {test_name}")
            
            # Look for AssertionError with JSON - handle multiline and various formats
            # The JSON might be spread across multiple lines
            json_patterns = [
                # Pattern for full AssertionError with JSON
                r'AssertionError: Total of \d+ vulnerabilities need to be fixed[^\{]*(\{.*?\})',
                # Pattern for just the JSON part (in case AssertionError is on different line)
                r'(\{"[^"]*": \[.*?\}\}*\])',
                # More flexible JSON pattern
                r'(\{[^{}]*(?:"vulnerability_id"[^{}]*)+\})',
            ]
            
            for i, pattern in enumerate(json_patterns):
                self.logger.info(f"   🔍 Trying ECR JSON pattern {i+1}...")
                matches = re.finditer(pattern, logs, re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    try:
                        json_str = match.group(1)
                        self.logger.info(f"   📋 Found JSON: {json_str[:100]}...")
                        
                        # Parse the JSON
                        vuln_data = json.loads(json_str)
                        self.logger.info(f"   ✅ Successfully parsed JSON with {len(vuln_data)} packages")
                        
                        # Extract vulnerabilities from each package
                        for package_name, package_vulns in vuln_data.items():
                            if isinstance(package_vulns, list):
                                self.logger.info(f"   📦 Processing package: {package_name} ({len(package_vulns)} vulns)")
                                for vuln in package_vulns:
                                    if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                                        cve_id = vuln['vulnerability_id']
                                        description = vuln.get('description', f'Vulnerability {cve_id}')
                                        severity = vuln.get('severity', 'UNKNOWN')
                                        
                                        issue = SecurityIssue(
                                            package=package_name,
                                            vulnerability_id=cve_id,
                                            description=description,
                                            severity=severity,
                                            test_type='os_scan',
                                            fix_action=""
                                        )
                                        issues.append(issue)
                                        self.logger.info(f"   📋 Found ECR vulnerability: {cve_id} in {package_name} ({severity})")
                        
                        if issues:  # If we found issues, break from trying more patterns
                            break
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"   ❌ JSON parse failed: {e}")
                        continue
                
                if issues:  # If we found issues, break from trying more patterns
                    break
                    
        elif 'test_safety_report_file' in test_name or 'safety' in test_name.lower():
            # Parse PyScan issues (SAFETY_REPORT format)
            self.logger.info(f"📊 Parsing PyScan vulnerabilities from {test_name}")
            
            # Handle the extra characters before SAFETY_REPORT (like sssssSAFETY_REPORT)
            # Make the pattern more flexible to handle prefixes
            safety_patterns = [
                # More flexible pattern that allows for prefixes
                r'.*?SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\][^\[]*\[installed: ([^\]]+)\][^\[]*\[vulnerabilities: \[.*?vulnerability_id=\'([^\']+)\'.*?advisory=\'([^\']*?)\'',
                # Alternative pattern in case the first doesn't work
                r'SAFETY_REPORT \(FAILED\).*?pkg: ([^\]]+).*?vulnerability_id=\'([^\']+)\'.*?advisory=\'([^\']*?)\'',
                # Even simpler pattern
                r'pkg: ([^\]]+).*?vulnerability_id=\'([^\']+)\'.*?advisory=\'([^\']*?)\'',
            ]
            
            for i, pattern in enumerate(safety_patterns):
                self.logger.info(f"   🔍 Trying PyScan pattern {i+1}...")
                matches = re.finditer(pattern, logs, re.DOTALL)
                
                match_count = 0
                for match in matches:
                    match_count += 1
                    groups = match.groups()
                    self.logger.info(f"   ✅ PyScan match {match_count}: {len(groups)} groups")
                    
                    if len(groups) >= 3:
                        package = groups[0].strip()
                        vuln_id = groups[-2].strip()  # vulnerability_id is usually second to last
                        advisory = groups[-1].strip()  # advisory is usually last
                        
                        # Handle case where we have installed version as well
                        if len(groups) == 4:
                            # package, installed, vuln_id, advisory
                            package = groups[0].strip()
                            vuln_id = groups[2].strip()
                            advisory = groups[3].strip()
                        
                        issue = SecurityIssue(
                            package=package,
                            vulnerability_id=vuln_id,
                            description=advisory,
                            severity="UNKNOWN",
                            test_type='py_scan',
                            fix_action=""
                        )
                        issues.append(issue)
                        self.logger.info(f"   📋 Found PyScan vulnerability: {vuln_id} in {package}")
                
                if match_count > 0:
                    self.logger.info(f"   ✅ PyScan pattern {i+1} found {match_count} matches")
                    break
                else:
                    self.logger.info(f"   ❌ PyScan pattern {i+1} no matches")
        
        self.logger.info(f"📊 Total extracted {len(issues)} security issues from {test_name}")
        
        # Debug: If no issues found, show what we're working with
        if not issues:
            self.logger.info(f"🔍 DEBUG: No issues found. Let me check specific content...")
            
            # Look for SAFETY_REPORT lines specifically
            safety_lines = [line for line in logs.split('\n') if 'SAFETY_REPORT' in line and 'FAILED' in line]
            if safety_lines:
                self.logger.info(f"   Found {len(safety_lines)} SAFETY_REPORT lines:")
                for i, line in enumerate(safety_lines[:2]):  # Show first 2
                    self.logger.info(f"   Line {i+1}: {line[:300]}...")
            
            # Look for AssertionError lines specifically  
            assertion_lines = [line for line in logs.split('\n') if 'AssertionError' in line and 'vulnerabilities' in line]
            if assertion_lines:
                self.logger.info(f"   Found {len(assertion_lines)} AssertionError lines:")
                for i, line in enumerate(assertion_lines[:2]):  # Show first 2
                    self.logger.info(f"   Line {i+1}: {line[:300]}...")
        
        return issues

    def create_dynamic_allowlist_fixes(self, issues: List[SecurityIssue]) -> tuple:
        """Create allowlist fixes dynamically based on extracted issues - CORRECTED FORMAT"""
        
        os_fixes = []
        py_fixes = []
        
        for issue in issues:
            if issue.test_type == 'os_scan':
                # Create OS scan allowlist entry (ECR Enhanced Scan format)
                os_fix = {
                    'vulnerability_id': issue.vulnerability_id,
                    'package': issue.package,
                    'description': issue.description
                }
                os_fixes.append(os_fix)
                self.logger.info(f"   📝 Created OS allowlist entry for {issue.vulnerability_id} ({issue.package})")
                
            elif issue.test_type == 'py_scan':
                # Create Python scan allowlist entry (simple key-value format)
                py_fix = {
                    'vulnerability_id': issue.vulnerability_id,
                    'description': issue.description
                }
                py_fixes.append(py_fix)
                self.logger.info(f"   📝 Created PyScan allowlist entry for {issue.vulnerability_id}")
        
        return os_fixes, py_fixes

    def get_dockerfile_content(self, container_type: str, device_type: str) -> str:
        """Get current Dockerfile content"""
        major_minor = '.'.join(self.current_version.split('.')[:2])
        
        if device_type == 'cpu':
            dockerfile_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
        else:
            py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
            cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
            if cuda_dirs:
                dockerfile_path = cuda_dirs[0] / "Dockerfile.gpu"
            else:
                return ""
                
        if dockerfile_path.exists():
            return dockerfile_path.read_text()
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
                # Apply OS scan allowlist fixes (ECR Enhanced Scan format - using ORIGINAL data)
                if os_fixes:
                    current_os_allowlist = self.get_allowlist_content(container_type, device_type, 'os_scan')
                    
                    self.logger.info("="*80)
                    self.logger.info(f"🔍 OS ALLOWLIST ENTRIES BEING ADDED ({container_type}/{device_type})")
                    self.logger.info("="*80)
                    
                    for fix in os_fixes:
                        vuln_id = fix['vulnerability_id']
                        package = fix['package']
                        
                        # Use the original vulnerability data if available - PRESERVE EXACT FORMAT
                        if 'original_vulnerability_data' in fix and fix['original_vulnerability_data']:
                            # Use the complete original vulnerability data exactly as extracted
                            allowlist_entry = fix['original_vulnerability_data'].copy()
                            # Only change the reason_to_ignore field
                            allowlist_entry['reason_to_ignore'] = "Security vulnerability allowlisted for AutoGluon DLC"
                            
                            self.logger.info(f"📋 Using ORIGINAL data format for {vuln_id}:")
                            # Log the first few fields to verify order
                            entry_keys = list(allowlist_entry.keys())[:5]
                            self.logger.info(f"   Field order: {entry_keys}...")
                        else:
                            # Fallback to basic structure with original field order
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
                        
                        # Group by package name (like your examples: "nltk": [...])
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
                
                # Apply Python scan allowlist fixes (simple key-value format)
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

    def debug_dockerfile_content(self, container_type: str, device_type: str) -> None:
        """Debug method to show Dockerfile content"""
        major_minor = '.'.join(self.current_version.split('.')[:2])
        
        if device_type == 'cpu':
            dockerfile_path = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3/Dockerfile.cpu"
        else:
            py3_dir = self.repo_dir / f"autogluon/{container_type}/docker/{major_minor}/py3"
            cuda_dirs = [d for d in py3_dir.iterdir() if d.is_dir() and d.name.startswith('cu')]
            if cuda_dirs:
                dockerfile_path = cuda_dirs[0] / "Dockerfile.gpu"
            else:
                self.logger.warning(f"No CUDA dirs found for {container_type}")
                return
        
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            lines = content.split('\n')
            
            self.logger.info(f"🔍 DEBUG: {dockerfile_path}")
            self.logger.info(f"📄 Total lines: {len(lines)}")
            self.logger.info(f"📄 File size: {len(content)} characters")
            
            # Show last 10 lines
            self.logger.info("📄 Last 10 lines:")
            for i, line in enumerate(lines[-10:], start=len(lines)-9):
                self.logger.info(f"   {i:3d}: {line}")
            
            # Check for security-related content
            security_lines = [i for i, line in enumerate(lines) if 'security' in line.lower()]
            if security_lines:
                self.logger.info(f"🔍 Found {len(security_lines)} lines with 'security':")
                for line_num in security_lines[-3:]:  # Show last 3
                    self.logger.info(f"   {line_num:3d}: {lines[line_num]}")
        else:
            self.logger.error(f"❌ Dockerfile not found: {dockerfile_path}")
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
            if test.get('check_run_id'):
                logs = self.get_test_logs(test['check_run_id'])
            
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
            result = subprocess.run(["git", "diff", "--quiet"], capture_output=True)
            if result.returncode == 0:
                self.logger.info("ℹ️ No changes to commit")
                return True
            
            # Show what changes will be committed
            self.logger.info("📋 Changes to be committed:")
            diff_result = subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True)
            if diff_result.stdout:
                for file in diff_result.stdout.strip().split('\n'):
                    self.logger.info(f"   📝 Modified: {file}")
            
            # Show a preview of the changes
            self.logger.info("\n📄 Preview of changes:")
            diff_preview = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True)
            if diff_preview.stdout:
                for line in diff_preview.stdout.strip().split('\n'):
                    self.logger.info(f"   {line}")
            
            # Show the commit message
            self.logger.info(f"\n💬 Commit message: {commit_message}")
            
            # Ask for user confirmation
            self.logger.info("\n" + "="*60)
            self.logger.info("🤔 CONFIRMATION REQUIRED:")
            self.logger.info("="*60)
            
            while True:
                try:
                    user_input = input("Do you want to commit and push these AI-recommended changes? (y/n): ").strip().lower()
                    
                    if user_input == 'y' or user_input == 'yes':
                        self.logger.info("✅ User confirmed - proceeding with commit and push")
                        break
                    elif user_input == 'n' or user_input == 'no':
                        self.logger.info("❌ User cancelled - aborting commit and push")
                        return False
                    else:
                        print("Please enter 'y' for yes or 'n' for no")
                        continue
                        
                except (EOFError, KeyboardInterrupt):
                    self.logger.info("\n❌ User interrupted - aborting commit and push")
                    return False
            
            # Add all changes
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # Push to branch
            subprocess.run(["git", "push", "origin", self.branch_name], check=True)
            
            self.logger.info(f"✅ Successfully committed and pushed: {commit_message}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to commit and push: {e}")
            return False
        finally:
            os.chdir(original_dir)

    def commit_and_push_changes_with_preview(self, commit_message: str) -> bool:
        """Alternative version with detailed diff preview"""
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            
            # Check if there are changes
            result = subprocess.run(["git", "diff", "--quiet"], capture_output=True)
            if result.returncode == 0:
                self.logger.info("ℹ️ No changes to commit")
                return True
            
            # Show detailed changes
            self.logger.info("📋 Detailed changes to be committed:")
            self.logger.info("="*60)
            
            # Show file names and line counts
            diff_stat = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True)
            if diff_stat.stdout:
                self.logger.info("📊 Change summary:")
                for line in diff_stat.stdout.strip().split('\n'):
                    self.logger.info(f"   {line}")
            
            # Show first few lines of each changed file
            diff_files = subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True)
            if diff_files.stdout:
                self.logger.info("\n📝 Preview of changes:")
                for file in diff_files.stdout.strip().split('\n')[:3]:  # Show max 3 files
                    self.logger.info(f"\n   📄 {file}:")
                    file_diff = subprocess.run(["git", "diff", "--", file], capture_output=True, text=True)
                    if file_diff.stdout:
                        # Show first 10 lines of diff
                        diff_lines = file_diff.stdout.split('\n')[:10]
                        for line in diff_lines:
                            if line.startswith('+'):
                                self.logger.info(f"   🟢 {line}")
                            elif line.startswith('-'):
                                self.logger.info(f"   🔴 {line}")
                            elif line.startswith('@@'):
                                self.logger.info(f"   📍 {line}")
                            else:
                                self.logger.info(f"      {line}")
                        
                        if len(file_diff.stdout.split('\n')) > 10:
                            self.logger.info("   ... (more changes)")
            
            # Show the commit message
            self.logger.info(f"\n💬 Commit message: {commit_message}")
            
            # Ask for user confirmation
            self.logger.info("\n" + "="*60)
            self.logger.info("🤔 CONFIRMATION REQUIRED:")
            self.logger.info("="*60)
            
            while True:
                try:
                    user_input = input("Do you want to commit and push these changes? (y/n): ").strip().lower()
                    
                    if user_input == 'y':
                        self.logger.info("✅ User confirmed - proceeding with commit and push")
                        break
                    elif user_input == 'n':
                        self.logger.info("❌ User cancelled - aborting commit and push")
                        return False
                    else:
                        print("Please enter 'y' for yes or 'n' for no")
                        continue
                        
                except (EOFError, KeyboardInterrupt):
                    self.logger.info("\n❌ User interrupted - aborting commit and push")
                    return False
            
            # Add all changes
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit changes
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # Push to branch
            subprocess.run(["git", "push", "origin", self.branch_name], check=True)
            
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
                # FIXED: Remove sort_keys=True to preserve original field order
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
                            # This was a continuation line, might need to fix the previous line
                            if new_lines and new_lines[-1].strip().endswith(' \\'):
                                # Previous line also had continuation, check if we need to remove it
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
    def extract_version_constraint_from_vulnerability(self, vuln_data: Dict, package_name: str) -> str:
        """Extract specific version constraint from vulnerability remediation data and description"""
        try:
            # First check remediation recommendation
            remediation = vuln_data.get('remediation', {})
            recommendation = remediation.get('recommendation', {})
            recommendation_text = recommendation.get('text', '')
            
            if recommendation_text and recommendation_text != "None Provided":
                # Look for version patterns in recommendation text
                version_patterns = [
                    # Pattern: "upgrade to version X.X.X or later"
                    r'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?: or later)?',
                    # Pattern: "fixed in version X.X.X"
                    r'fixed in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                    # Pattern: "version >= X.X.X"
                    r'version >= ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                    # Pattern: just version numbers
                    r'([0-9]+\.[0-9]+(?:\.[0-9]+)?)'
                ]
                
                for pattern in version_patterns:
                    match = re.search(pattern, recommendation_text, re.IGNORECASE)
                    if match:
                        fixed_version = match.group(1)
                        self.logger.info(f"📋 Found fixed version in remediation for {package_name}: {fixed_version}")
                        return f">={fixed_version}"
            
            # Check description field for version information
            description = vuln_data.get('description', '')
            if description:
                description_constraint = self.extract_version_from_description(description, package_name, vuln_data)
                if description_constraint and description_constraint != "latest":
                    return description_constraint
            
            # Check if there's version information in the package details
            package_details = vuln_data.get('package_details', {})
            current_version = package_details.get('version', '')
            
            if current_version:
                # Try to increment the version slightly for a safer upgrade
                try:
                    version_parts = current_version.split('.')
                    if len(version_parts) >= 2:
                        major = int(version_parts[0])
                        minor = int(version_parts[1])
                        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
                        
                        # For security fixes, usually increment patch version
                        next_patch_version = f"{major}.{minor}.{patch + 1}"
                        return f">={next_patch_version}"
                except (ValueError, IndexError):
                    pass
            
            # Default fallback
            return "latest"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract version constraint for {package_name}: {e}")
            return "latest"

    def extract_version_from_description(self, description: str, package_name: str, vuln_data: Dict) -> str:
        """Extract version constraint from vulnerability description using pattern matching and AI"""
        try:
            self.logger.info(f"🔍 Analyzing description for version info for {package_name}")
            
            # First try rule-based pattern matching
            rule_based_constraint = self.extract_version_from_description_patterns(description, package_name)
            if rule_based_constraint and rule_based_constraint != "latest":
                self.logger.info(f"📋 Found version constraint via patterns for {package_name}: {rule_based_constraint}")
                return rule_based_constraint
            
            # If pattern matching fails, use AI for complex cases
            ai_constraint = self.extract_version_from_description_with_ai(description, package_name, vuln_data)
            if ai_constraint and ai_constraint != "latest":
                self.logger.info(f"🤖 Found version constraint via AI for {package_name}: {ai_constraint}")
                return ai_constraint
            
            return "latest"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract version from description for {package_name}: {e}")
            return "latest"

    def extract_version_from_description_patterns(self, description: str, package_name: str) -> str:
        """Extract version constraint using rule-based pattern matching"""
        try:
            # Pattern 1: "upgrade to [package] version X.X.X or later"
            upgrade_patterns = [
                rf'upgrade to {re.escape(package_name)} version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?: or later)?',
                rf'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?: or later)?',
                rf'{re.escape(package_name)} version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?: or later)?',
                r'version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?: or later)?'
            ]
            
            for pattern in upgrade_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    self.logger.info(f"📋 Pattern matched upgrade version for {package_name}: {version}")
                    return f">={version}"
            
            # Pattern 2: "prior to version X.X.X" (means we need >= X.X.X)
            prior_patterns = [
                rf'{re.escape(package_name)} prior to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'prior to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'{re.escape(package_name)} before version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'before version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)'
            ]
            
            for pattern in prior_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    self.logger.info(f"📋 Pattern matched 'prior to' version for {package_name}: {version}")
                    return f">={version}"
            
            # Pattern 3: "fixed in version X.X.X"
            fixed_patterns = [
                rf'fixed in {re.escape(package_name)} ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'fixed in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'patched in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)'
            ]
            
            for pattern in fixed_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    self.logger.info(f"📋 Pattern matched 'fixed in' version for {package_name}: {version}")
                    return f">={version}"
            
            # Pattern 4: "through version X.X.X" (means we need > X.X.X)
            through_patterns = [
                rf'{re.escape(package_name)} through ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                r'through version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)'
            ]
            
            for pattern in through_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    next_version = self.increment_version(version, increment_patch=True)
                    self.logger.info(f"📋 Pattern matched 'through' version for {package_name}: {version} -> {next_version}")
                    return f">={next_version}"
            
            return "latest"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pattern matching failed for {package_name}: {e}")
            return "latest"

    def extract_version_from_description_with_ai(self, description: str, package_name: str, vuln_data: Dict) -> str:
        """Use AI to extract version constraint from complex vulnerability descriptions"""
        try:
            if not hasattr(self, 'llm'):
                self.logger.warning("⚠️ AI not available for version extraction")
                return "latest"
            
            # Get current version from vulnerability data
            current_version = vuln_data.get('package_details', {}).get('version', 'unknown')
            
            # Create AI prompt for version extraction
            version_extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a security vulnerability analyzer. Your job is to extract version upgrade information from vulnerability descriptions.

    TASK: Extract the minimum safe version that fixes the vulnerability.

    RULES:
    1. Look for phrases like "upgrade to version X.X.X", "fixed in version X.X.X", "prior to version X.X.X"
    2. If description says "prior to X.X.X" or "before X.X.X", the safe version is X.X.X
    3. If description says "upgrade to X.X.X or later", the safe version is X.X.X
    4. If description says "fixed in X.X.X", the safe version is X.X.X
    5. If description says "through X.X.X", the safe version is the next patch version after X.X.X

    OUTPUT FORMAT:
    Return ONLY the version number (e.g., "5.8.0") or "none" if no clear version can be extracted.
    Do not include ">=" or other operators, just the version number.
    Do not include any explanation or additional text."""),
                ("human", """Package: {package_name}
    Current Version: {current_version}
    Vulnerability Description: {description}

    Extract the minimum safe version:""")
            ])
            
            version_chain = version_extraction_prompt | self.llm
            
            self.logger.info(f"🤖 Using AI to extract version from description for {package_name}")
            ai_response = version_chain.invoke({
                "package_name": package_name,
                "current_version": current_version,
                "description": description
            })
            
            # Parse AI response
            ai_version = str(ai_response).strip().lower()
            
            if ai_version == "none" or not ai_version:
                self.logger.info(f"🤖 AI found no version info for {package_name}")
                return "latest"
            
            # Validate the extracted version format
            if re.match(r'^[0-9]+\.[0-9]+(?:\.[0-9]+)?$', ai_version):
                self.logger.info(f"🤖 AI extracted version for {package_name}: {ai_version}")
                return f">={ai_version}"
            else:
                self.logger.warning(f"🤖 AI returned invalid version format for {package_name}: {ai_version}")
                return "latest"
            
        except Exception as e:
            self.logger.warning(f"⚠️ AI version extraction failed for {package_name}: {e}")
            return "latest"
    def check_constraint_compatibility(self, package_name: str, constraint1: str, constraint2: str, current_version: str) -> bool:
        """Check if two version constraints are compatible"""
        try:
            # Get available versions to test against
            available_versions = self.get_available_versions_pip_index(package_name)
            if not available_versions:
                return False
            
            # Test if any versions satisfy both constraints
            compatible_versions = []
            
            for version in available_versions[:10]:  # Test first 10 versions
                satisfies_1 = self.version_satisfies_constraint(version, constraint1)
                satisfies_2 = self.version_satisfies_constraint(version, constraint2)
                
                if satisfies_1 and satisfies_2:
                    compatible_versions.append(version)
            
            is_compatible = len(compatible_versions) > 0
            
            if is_compatible:
                self.logger.info(f"✅ Constraints compatible for {package_name}: found {len(compatible_versions)} versions")
                self.logger.info(f"   Compatible versions: {compatible_versions[:3]}")
            else:
                self.logger.warning(f"❌ Constraints incompatible for {package_name}: no versions satisfy both")
                
                # Show what each constraint would allow
                versions_1 = [v for v in available_versions[:5] if self.version_satisfies_constraint(v, constraint1)]
                versions_2 = [v for v in available_versions[:5] if self.version_satisfies_constraint(v, constraint2)]
                
                self.logger.warning(f"   Constraint '{constraint1}' allows: {versions_1}")
                self.logger.warning(f"   Constraint '{constraint2}' allows: {versions_2}")
            
            return is_compatible
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check constraint compatibility for {package_name}: {e}")
            return False
    def get_optimal_version_constraint(self, package_name: str, vuln_data: Dict, all_logs: str, vulnerability_type: str = 'os_scan') -> str:
        """Get the best version constraint from multiple sources - SEPARATED BY VULNERABILITY TYPE"""
        
        if vulnerability_type == 'py_scan':
            # For Python scan vulnerabilities, only use Safety spec constraints
            self.logger.info(f"🔍 Getting Python scan constraint for {package_name}")
            log_constraint = self.parse_version_constraints_from_logs(all_logs, package_name)
            
            if log_constraint and log_constraint != "latest":
                self.logger.info(f"📋 Using Safety spec constraint for {package_name}: {log_constraint}")
                return log_constraint
            else:
                self.logger.info(f"📋 No Safety spec found for {package_name}, using latest")
                return "latest"
        
        else:
            # For OS scan vulnerabilities, only use description parsing (no Safety specs)
            self.logger.info(f"🔍 Getting OS scan constraint for {package_name} (description parsing only)")
            vuln_constraint = self.extract_version_constraint_from_vulnerability(vuln_data, package_name)
            
            if vuln_constraint and vuln_constraint != "latest":
                self.logger.info(f"📋 Using vulnerability description constraint for {package_name}: {vuln_constraint}")
                return vuln_constraint
            else:
                self.logger.info(f"📋 No description constraint found for {package_name}, using latest")
                return "latest"
    def wait_for_security_tests_to_complete(self, pr_number: int, max_wait_minutes: int = 180) -> bool:
        """
        Wait for security tests to start and complete using both GraphQL + REST API.
        GraphQL for AutoGluon security tests (2 tests) + REST API for other security tests.
        """
        self.logger.info(f"⏳ Waiting for security tests to complete on PR #{pr_number} (using GraphQL + REST API)...")
        self.logger.info(f"Will wait up to {max_wait_minutes} minutes for security tests to finish")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval_seconds = 20  # Check every minute
        
        consecutive_stable_checks = 0
        required_stable_checks = 3  # Need 3 consecutive stable results
        
        while (time.time() - start_time) < max_wait_seconds:
            try:
                # Get all security tests via combined approach
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
        return True  # Proceed anyway after timeout

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
                                'check_run_id': None,  # Status contexts don't have check run IDs
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
        Main agent execution - DYNAMIC VERSION that handles any vulnerabilities
        """
        return self.run_deterministic_security_analysis()
    # REPLACE the preprocess_logs_for_ai method with this ENHANCED DEBUG version:

    # FIX 1: Better OS Scan JSON Extraction
# REPLACE your preprocess_logs_for_ai method with this improved version:

    def preprocess_logs_for_ai(self, logs: str) -> str:
        """Preprocess logs and preserve original vulnerability data for exact allowlist formatting"""
        self.logger.info("🔧 Preprocessing logs for better AI analysis...")
        
        processed_logs = logs
        
        # Store original vulnerability data for later use
        self.original_vulnerability_data = {}
        
        # Find ALL complete JSON blocks - FIXED TO PROCESS ALL BLOCKS
        lines = logs.split('\n')
        processed_blocks = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'Total of' in line and 'vulnerabilities need to be fixed' in line and ':' in line:
                self.logger.info(f"🔍 Found vulnerability line: {line[:100]}...")
                
                # Look for JSON starting in the next few lines
                json_content = ""
                for j in range(i + 1, min(i + 20, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith('{'):
                        # Found start of JSON, now collect the complete JSON block
                        brace_count = 0
                        json_lines = []
                        
                        for k in range(j, len(lines)):
                            check_line = lines[k]
                            json_lines.append(check_line)
                            
                            # Count braces to find complete JSON
                            brace_count += check_line.count('{') - check_line.count('}')
                            
                            # If we've closed all braces, we have complete JSON
                            if brace_count == 0 and any(char in check_line for char in '}'):
                                json_content = '\n'.join(json_lines)
                                i = k  # Skip past this block
                                break
                            
                            # Safety check - don't go too far
                            if k - j > 50:
                                break
                        break
                
                if json_content:
                    try:
                        # Clean up the JSON content
                        json_content = json_content.strip()
                        if '{' in json_content:
                            json_start = json_content.find('{')
                            json_content = json_content[json_start:]
                        
                        self.logger.info(f"🔍 Attempting to parse complete JSON block ({len(json_content)} chars)")
                        
                        parsed_json = json.loads(json_content)
                        
                        # Store original vulnerability data for each package
                        for package_name, package_vulns in parsed_json.items():
                            if isinstance(package_vulns, list):
                                # If package already exists, extend the list instead of overwriting
                                if package_name in self.original_vulnerability_data:
                                    self.original_vulnerability_data[package_name].extend(package_vulns)
                                else:
                                    self.original_vulnerability_data[package_name] = package_vulns
                                
                                self.logger.info(f"📋 Stored original data for {package_name}: {len(package_vulns)} vulnerabilities")
                        
                        processed_blocks += 1
                        self.logger.info(f"✅ Successfully processed vulnerability block {processed_blocks} with {len(parsed_json)} packages")
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"❌ Could not parse JSON block: {e}")
            
            i += 1
    
        
        # Extract Python scan data (keep existing logic)
        self.logger.info("🔍 Searching for Python scan vulnerability patterns...")
        
        safety_pattern = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'(?:.*?advisory=\'([^\']*?)(?:\'|\.\.\.|$))?'
        safety_matches = list(re.finditer(safety_pattern, logs, re.DOTALL))
        
        py_vulnerabilities = []
        seen_vulnerabilities = set()
        
        for match in safety_matches:
            groups = match.groups()
            package = groups[0].strip()
            vuln_id = groups[1].strip()
            advisory = groups[2].strip() if groups[2] else f'Security vulnerability in {package}'
            
            unique_key = (package, vuln_id)
            if unique_key not in seen_vulnerabilities:
                seen_vulnerabilities.add(unique_key)
                py_vulnerabilities.append({
                    "package": package,
                    "vulnerability_id": vuln_id,
                    "description": advisory
                })
        
        if py_vulnerabilities:
            processed_section = f"""

    === PYTHON SCAN VULNERABILITIES (Safety Report) ===
    {json.dumps(py_vulnerabilities, indent=2)}
    === END PYTHON SCAN VULNERABILITIES ===
    """
            processed_logs += processed_section
            self.logger.info(f"✅ Extracted {len(py_vulnerabilities)} unique Python vulnerabilities")
        
        self.logger.info(f"🔧 Preprocessing complete. Stored original data for {len(self.original_vulnerability_data)} packages")
        return processed_logs
    def parse_ai_response_with_fallback(self, ai_response_text: str) -> Dict:
        """Parse AI response with fallback for non-JSON responses - TARGETED FIX"""
        self.logger.info("🤖 Parsing AI response...")
        
        # First try to parse as JSON
        try:
            if isinstance(ai_response_text, dict):
                return ai_response_text
            
            # Try to parse the full response as JSON
            return json.loads(ai_response_text)
        except (json.JSONDecodeError, TypeError):
            self.logger.info("🤖 Full response not JSON, extracting JSON block...")
            
            # Extract JSON block from AI response - IMPROVED EXTRACTION
            try:
                response_str = str(ai_response_text)
                
                # Method 1: Find JSON between { and } with proper brace counting
                start_pos = response_str.find('{')
                if start_pos != -1:
                    brace_count = 0
                    end_pos = start_pos
                    
                    for i in range(start_pos, len(response_str)):
                        char = response_str[i]
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if brace_count == 0:  # Found complete JSON
                        json_text = response_str[start_pos:end_pos]
                        self.logger.info(f"🤖 Extracted JSON block ({len(json_text)} chars)")
                        
                        # Clean up any potential issues
                        json_text = json_text.strip()
                        
                        # Show a preview of what we're trying to parse
                        self.logger.info(f"🤖 JSON preview: {json_text[:200]}...")
                        self.logger.info(f"🤖 JSON ending: ...{json_text[-200:]}")
                        
                        parsed = json.loads(json_text)
                        self.logger.info(f"✅ Successfully parsed JSON with keys: {list(parsed.keys())}")
                        return parsed
                        
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ JSON extraction failed: {e}")
                # Show the problematic area
                error_pos = getattr(e, 'pos', 0)
                if 'json_text' in locals():
                    problematic_area = json_text[max(0, error_pos-100):error_pos+100]
                    self.logger.error(f"Problem area: {problematic_area}")
            except Exception as e:
                self.logger.error(f"❌ Unexpected error during JSON extraction: {e}")
            
            # Method 2: Try to extract using regex patterns
            self.logger.info("🔄 Trying regex extraction methods...")
            try:
                # Pattern for our specific JSON structure
                pattern = r'\{\s*"os_scan_issues".*?\}\s*(?=\n\n|\nKey|\n[A-Z]|$)'
                match = re.search(pattern, response_str, re.DOTALL)
                
                if match:
                    json_candidate = match.group(0)
                    self.logger.info(f"🤖 Regex found JSON candidate ({len(json_candidate)} chars)")
                    
                    parsed = json.loads(json_candidate)
                    self.logger.info(f"✅ Regex extraction successful with keys: {list(parsed.keys())}")
                    return parsed
                    
            except Exception as e:
                self.logger.error(f"❌ Regex extraction failed: {e}")
            
            # Fallback to manual parsing
            self.logger.warning("🔄 Falling back to manual vulnerability extraction...")
            return self.extract_vulnerabilities_from_ai_text(response_str)
    # ADD helper methods for better context extraction:



    def extract_vulnerabilities_from_ai_text(self, ai_text: str) -> Dict:
        """Extract vulnerability info from AI text when JSON parsing fails"""
        self.logger.info("🔍 Manually extracting vulnerabilities from AI text...")
        
        os_issues = []
        py_issues = []
        
        # Split into lines and look for vulnerability patterns
        lines = ai_text.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            # Detect sections
            if '"os_scan_issues"' in line:
                current_section = 'os'
                continue
            elif '"py_scan_issues"' in line:
                current_section = 'py'
                continue
            
            # Extract vulnerability data from individual lines
            package_match = re.search(r'"package":\s*"([^"]+)"', line)
            vuln_id_match = re.search(r'"vulnerability_id":\s*"([^"]+)"', line)
            desc_match = re.search(r'"description":\s*"([^"]+)"', line)
            sev_match = re.search(r'"severity":\s*"([^"]+)"', line)
            
            if package_match and vuln_id_match:
                vulnerability = {
                    'package': package_match.group(1),
                    'vulnerability_id': vuln_id_match.group(1),
                    'description': desc_match.group(1) if desc_match else f'Vulnerability {vuln_id_match.group(1)}',
                    'severity': sev_match.group(1) if sev_match else 'UNKNOWN'
                }
                
                if current_section == 'os':
                    os_issues.append(vulnerability)
                    self.logger.info(f"🔍 Manual extraction found OS: {vulnerability['vulnerability_id']} in {vulnerability['package']}")
                elif current_section == 'py':
                    py_issues.append(vulnerability)
                    self.logger.info(f"🔍 Manual extraction found Python: {vulnerability['vulnerability_id']} in {vulnerability['package']}")
        
        # Also try to extract from the structured part we can see
        cve_pattern = r'(CVE-\d{4}-\d{4,})[^a-zA-Z]*([a-zA-Z][a-zA-Z0-9_-]*)'
        cve_matches = re.findall(cve_pattern, ai_text)
        
        for cve_id, package in cve_matches:
            package = package.lower()
            # Don't duplicate if we already found it
            exists = any(issue['vulnerability_id'] == cve_id and issue['package'] == package for issue in os_issues)
            if not exists:
                os_issues.append({
                    'package': package,
                    'vulnerability_id': cve_id,
                    'description': f'Manual extraction: {cve_id} in {package}',
                    'severity': 'UNKNOWN'
                })
                self.logger.info(f"🔍 Manual pattern found: {cve_id} in {package}")
        
        self.logger.info(f"🔍 Manual extraction complete: {len(os_issues)} OS, {len(py_issues)} Python vulnerabilities")
        
        return {
            'os_scan_issues': os_issues,
            'py_scan_issues': py_issues,
            'dockerfile_fixes': [],
            'os_scan_allowlist_fixes': [],
            'py_scan_allowlist_fixes': [],
            'try_dockerfile_first': False,
            'severity_assessment': 'Manual extraction from AI text'
        }
    
    def version_satisfies_constraint(self, version: str, constraint: str) -> bool:
        """Check if a version satisfies a constraint (simple version comparison)"""
        try:
            # Remove constraint operators to get the version to compare against
            constraint_version = constraint.lstrip('<>=!')
            
            # Simple version comparison (assumes semantic versioning)
            v1_parts = [int(x) for x in version.split('.')]
            v2_parts = [int(x) for x in constraint_version.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            # Compare versions
            if constraint.startswith('<='):
                return v1_parts <= v2_parts
            elif constraint.startswith('>='):
                return v1_parts >= v2_parts
            elif constraint.startswith('<'):
                return v1_parts < v2_parts
            elif constraint.startswith('>'):
                return v1_parts > v2_parts
            elif constraint.startswith('=='):
                return v1_parts == v2_parts
            else:
                return False
                
        except (ValueError, IndexError):
            self.logger.warning(f"⚠️ Could not compare version {version} with constraint {constraint}")
            return False
    def extract_severity_from_context(self, text: str, cve: str, package: str) -> str:
        """Extract severity from surrounding context"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if cve in line and package in line.lower():
                # Get surrounding context
                start = max(0, i-3)
                end = min(len(lines), i+6)
                context = '\n'.join(lines[start:end]).upper()
                
                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    if severity in context:
                        return severity
        return 'UNKNOWN'

    def extract_description_from_context(self, text: str, cve: str, package: str) -> str:
        """Extract description from AI response"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if cve in line and package in line.lower():
                # Look for description in surrounding lines
                start = max(0, i-2)
                end = min(len(lines), i+5)
                
                for j in range(start, end):
                    desc_line = lines[j]
                    if 'description' in desc_line.lower() and ':' in desc_line:
                        desc_match = re.search(r'description[:\s]*(.+)', desc_line, re.IGNORECASE)
                        if desc_match:
                            return desc_match.group(1).strip()
                    
                    # Also look for lines that seem like descriptions
                    if len(desc_line) > 50 and any(word in desc_line.lower() for word in ['vulnerability', 'allows', 'remote', 'execution']):
                        return desc_line.strip()
        
        return f"AI-detected {cve} in {package}"

    def get_package_context_from_ai_response(self, text: str, package: str, cve: str) -> str:
        """Get context around a specific package mention"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if package in line.lower() and cve in line:
                start = max(0, i-5)
                end = min(len(lines), i+10)
                return '\n'.join(lines[start:end])
        return ""

    def filter_versions_by_constraint(self, versions: List[str], constraint: str) -> List[str]:
        """Filter versions list by constraint string"""
        try:
            matching_versions = []
            
            if constraint == "latest":
                return versions[:1] if versions else []
            
            constraints = [c.strip() for c in constraint.split(',')]
            
            for version in versions:
                satisfies_all = True
                
                for single_constraint in constraints:
                    if not self.version_satisfies_constraint(version, single_constraint):
                        satisfies_all = False
                        break
                
                if satisfies_all:
                    matching_versions.append(version)
            
            return matching_versions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not filter versions by constraint '{constraint}': {e}")
            return []
    # FIX 3: Update the analyze_security_logs_with_ai method to handle both issues
    # REPLACE your analyze_security_logs_with_ai method:

    def analyze_security_logs_with_ai(self, all_logs: str, dockerfile_content: str, os_allowlist: str, py_allowlist: str) -> Dict:
        """Use AI to analyze security logs - WITH DETAILED DEBUG OUTPUT"""
        self.logger.info("🤖 Using AI to analyze security logs and extract vulnerabilities...")
        
        try:
            # Truncate logs first
            truncated_logs = self.truncate_logs_for_ai(all_logs, max_chars=100000)
            
            # Preprocess the truncated logs
            processed_logs = self.preprocess_logs_for_ai(truncated_logs)
            
            # Prepare context for the AI
            analysis_input = {
                "version": self.current_version,
                "branch": self.branch_name,
                "security_logs": processed_logs,
                "dockerfile_content": dockerfile_content or "Not provided for extraction",
                "os_allowlist": os_allowlist or "{}",
                "py_allowlist": py_allowlist or "{}"
            }
            
            self.logger.info("🧠 AI is analyzing security vulnerabilities...")
            ai_response = self.chain.invoke(analysis_input)
            
            # DETAILED DEBUG OUTPUT
            self.logger.info("="*80)
            self.logger.info("🔍 DEBUG: RAW AI RESPONSE ANALYSIS")
            self.logger.info("="*80)
            self.logger.info(f"📋 Response type: {type(ai_response)}")
            self.logger.info(f"📋 Response length: {len(str(ai_response))} characters")
            
            # Show first and last parts of response
            response_str = str(ai_response)
            self.logger.info(f"📋 First 200 chars: {response_str[:200]}")
            self.logger.info(f"📋 Last 200 chars: {response_str[-200:]}")
            
            # Look for JSON patterns
            json_patterns = [
                r'\{.*?\}',  # Any JSON-like structure
                r'\{.*?"os_scan_issues".*?\}',  # Specific to our format
            ]
            
            for i, pattern in enumerate(json_patterns):
                matches = re.findall(pattern, response_str, re.DOTALL)
                self.logger.info(f"📋 JSON pattern {i+1} found {len(matches)} matches")
                if matches:
                    for j, match in enumerate(matches[:2]):  # Show first 2 matches
                        self.logger.info(f"   Match {j+1} length: {len(match)} chars")
                        self.logger.info(f"   Match {j+1} start: {match[:100]}...")
                        self.logger.info(f"   Match {j+1} end: ...{match[-100:]}")
            
            # Try different extraction methods
            self.logger.info("🔍 Trying different JSON extraction methods...")
            
            # Method 1: Direct JSON parse
            try:
                if isinstance(ai_response, dict):
                    self.logger.info("✅ Method 1: Already a dict")
                    parsed_response = ai_response
                else:
                    parsed_response = json.loads(str(ai_response))
                    self.logger.info("✅ Method 1: Direct JSON parse succeeded")
            except json.JSONDecodeError as e:
                self.logger.info(f"❌ Method 1: Direct JSON parse failed: {e}")
                parsed_response = None
            
            # Method 2: Extract JSON block
            if parsed_response is None:
                self.logger.info("🔍 Method 2: Extracting JSON block...")
                json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    self.logger.info(f"📋 Found JSON block: {len(json_text)} chars")
                    self.logger.info(f"📋 JSON starts with: {json_text[:150]}...")
                    self.logger.info(f"📋 JSON ends with: ...{json_text[-150:]}")
                    
                    try:
                        parsed_response = json.loads(json_text)
                        self.logger.info("✅ Method 2: JSON block parse succeeded")
                    except json.JSONDecodeError as e:
                        self.logger.info(f"❌ Method 2: JSON block parse failed: {e}")
                        # Show where the error occurred
                        error_pos = getattr(e, 'pos', 0)
                        self.logger.info(f"📋 Error around position {error_pos}: {json_text[max(0, error_pos-50):error_pos+50]}")
                else:
                    self.logger.info("❌ Method 2: No JSON block found")
            
            # Method 3: Multiple JSON blocks
            if parsed_response is None:
                self.logger.info("🔍 Method 3: Looking for multiple JSON blocks...")
                json_blocks = re.findall(r'\{[^{}]*(?:{[^{}]*}[^{}]*)*\}', response_str)
                self.logger.info(f"📋 Found {len(json_blocks)} potential JSON blocks")
                
                for i, block in enumerate(json_blocks):
                    self.logger.info(f"📋 Block {i+1}: {len(block)} chars, starts: {block[:100]}...")
                    try:
                        test_parse = json.loads(block)
                        if 'os_scan_issues' in test_parse or 'py_scan_issues' in test_parse:
                            parsed_response = test_parse
                            self.logger.info(f"✅ Method 3: Block {i+1} is our target JSON")
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Method 4: Line by line JSON reconstruction
            if parsed_response is None:
                self.logger.info("🔍 Method 4: Line by line JSON reconstruction...")
                lines = response_str.split('\n')
                json_lines = []
                in_json = False
                brace_count = 0
                
                for line_num, line in enumerate(lines):
                    if '{' in line and not in_json:
                        in_json = True
                        self.logger.info(f"📋 JSON starts at line {line_num}: {line.strip()}")
                    
                    if in_json:
                        json_lines.append(line)
                        brace_count += line.count('{') - line.count('}')
                        
                        if brace_count == 0:
                            self.logger.info(f"📋 JSON ends at line {line_num}: {line.strip()}")
                            break
                
                if json_lines:
                    reconstructed_json = '\n'.join(json_lines)
                    self.logger.info(f"📋 Reconstructed JSON: {len(reconstructed_json)} chars")
                    try:
                        parsed_response = json.loads(reconstructed_json)
                        self.logger.info("✅ Method 4: Reconstructed JSON parse succeeded")
                    except json.JSONDecodeError as e:
                        self.logger.info(f"❌ Method 4: Reconstructed JSON parse failed: {e}")
            
            # Final result
            if parsed_response:
                self.logger.info("="*80)
                self.logger.info("🔍 DEBUG: PARSED RESPONSE ANALYSIS")
                self.logger.info("="*80)
                self.logger.info(f"📋 Parsed response type: {type(parsed_response)}")
                self.logger.info(f"📋 Parsed response keys: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else 'Not a dict'}")
                
                if isinstance(parsed_response, dict):
                    for key, value in parsed_response.items():
                        if isinstance(value, list):
                            self.logger.info(f"📋 {key}: {len(value)} items")
                            for i, item in enumerate(value[:2]):  # Show first 2 items
                                self.logger.info(f"   Item {i+1}: {item}")
                        else:
                            self.logger.info(f"📋 {key}: {value}")
                
                self.logger.info("="*80)
                
                return parsed_response
            else:
                self.logger.error("❌ All JSON extraction methods failed")
                raise Exception("Could not extract valid JSON from AI response")
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"❌ AI analysis failed: {error_str}")
            self.logger.info("🔄 Falling back to enhanced rule-based parsing...")
            return self.enhanced_fallback_with_preprocessed_data(all_logs)

    def enhance_ai_response_with_original_data(self, ai_response: Dict) -> None:
        """Enhance AI response with original vulnerability data for exact allowlist formatting"""
        try:
            # Add original vulnerability data to OS scan issues for proper allowlist formatting
            for issue in ai_response.get('os_scan_issues', []):
                package = issue.get('package', '').lower()
                vuln_id = issue.get('vulnerability_id', '')
                
                # Look for original data
                if hasattr(self, 'original_vulnerability_data') and package in self.original_vulnerability_data:
                    for orig_vuln in self.original_vulnerability_data[package]:
                        if orig_vuln.get('vulnerability_id') == vuln_id:
                            issue['original_data'] = orig_vuln
                            self.logger.info(f"🔗 Enhanced AI result with original data for {vuln_id}")
                            break
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not enhance AI response with original data: {e}")
    def apply_ai_recommended_fixes(self, ai_response: Dict, container_type: str) -> bool:
        """Apply AI-recommended fixes intelligently"""
        self.logger.info(f"🤖 Applying AI-recommended fixes for {container_type}")
        
        try:
            success = True
            
            # Check if AI recommends trying Dockerfile fixes first
            try_dockerfile_first = ai_response.get('try_dockerfile_first', False)
            dockerfile_fixes = ai_response.get('dockerfile_fixes', [])
            os_allowlist_fixes = ai_response.get('os_scan_allowlist_fixes', [])
            py_allowlist_fixes = ai_response.get('py_scan_allowlist_fixes', [])
            
            self.logger.info(f"🧠 AI Strategy:")
            self.logger.info(f"   Try Dockerfile first: {try_dockerfile_first}")
            self.logger.info(f"   Dockerfile fixes: {len(dockerfile_fixes)}")
            self.logger.info(f"   OS allowlist fixes: {len(os_allowlist_fixes)}")
            self.logger.info(f"   Python allowlist fixes: {len(py_allowlist_fixes)}")
            
            vulnerability_mapping = {}
            
            # Apply fixes based on AI strategy
            if try_dockerfile_first and dockerfile_fixes:
                self.logger.info("🔧 AI recommends trying Dockerfile fixes first...")
                dockerfile_success, vulnerability_mapping = self.apply_dockerfile_fixes(container_type, dockerfile_fixes)
                
                if dockerfile_success:
                    self.logger.info("✅ Dockerfile fixes applied - will commit and test")
                    return True
                else:
                    self.logger.warning("⚠️ Dockerfile fixes failed, proceeding to allowlist...")
            
            # Apply allowlist fixes (either as primary strategy or fallback)
            if os_allowlist_fixes or py_allowlist_fixes:
                self.logger.info("📝 Applying AI-recommended allowlist fixes...")
                allowlist_success = self.apply_allowlist_fixes(container_type, os_allowlist_fixes, py_allowlist_fixes)
                
                if allowlist_success:
                    self.logger.info("✅ AI allowlist fixes applied")
                    return True
                else:
                    self.logger.error("❌ AI allowlist fixes failed")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply AI-recommended fixes: {e}")
            return False

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
                
                # Find the AutoGluon installation line
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
                
                # Group fixes by install method with deduplication
                pip_packages_dict = {}  # Use dict to avoid duplicates
                
                for fix in fixes:
                    if fix['type'] == 'update_package':
                        package = fix['package']
                        version_constraint = fix.get('version', 'latest')
                        install_method = fix.get('install_method', 'pip')
                        vulnerability_id = fix.get('vulnerability_id', '')
                        
                        if vulnerability_id:
                            vulnerability_mapping[vulnerability_id] = package
                        
                        # Only handle pip packages (skip apt packages for this approach)
                        if install_method == 'pip':
                            # Check if this package already exists in the Dockerfile
                            package_already_exists = False
                            for line in lines:
                                if f"pip install" in line and f'"{package}' in line:
                                    self.logger.info(f"⚠️ Package {package} already appears to be installed in Dockerfile")
                                    package_already_exists = True
                                    break
                            
                            if not package_already_exists:
                                # Handle pip package deduplication
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
                
                # Convert pip packages dict to proper format
                pip_install_lines = []
                for package, version_constraint in pip_packages_dict.items():
                    if version_constraint and version_constraint != 'latest':
                        # Handle different constraint formats
                        if any(op in version_constraint for op in ['>=', '<=', '==', '!=', '<', '>', ',']):
                            # Already has constraint operators
                            pip_install_lines.append(f'&& pip install --no-cache-dir "{package}{version_constraint}"')
                        else:
                            # Just a version number, use ==
                            pip_install_lines.append(f'&& pip install --no-cache-dir "{package}=={version_constraint}"')
                    else:
                        # Use upgrade for latest
                        pip_install_lines.append(f'&& pip install --no-cache-dir "{package}" --upgrade')
                        self.logger.warning(f"⚠️ Using --upgrade for {package} - could be risky!")
                
                # Insert pip packages after the AutoGluon installation line
                if pip_install_lines:
                    self.logger.info(f"📝 Inserting {len(pip_install_lines)} pip install lines after AutoGluon installation")
                    
                    # Find the insertion point (after the AutoGluon line but before any non-pip lines)
                    insert_index = autogluon_line_index + 1
                    
                    # Add each pip install as a continuation line with proper formatting
                    for j, pip_line in enumerate(pip_install_lines):
                        # Add backslash to continue the command
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
    # REPLACE the fallback_rule_based_analysis method with this FIXED version:
    def enhanced_fallback_with_preprocessed_data(self, logs: str) -> Dict:
        """Enhanced fallback that preserves original vulnerability data"""
        self.logger.info("🔄 Using enhanced fallback with original vulnerability data...")
        
        os_issues = []
        py_issues = []
        
        # Get original vulnerability data if available
        if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
            self.logger.info(f"✅ Using stored original vulnerability data for {len(self.original_vulnerability_data)} packages")
            
            for package_name, package_vulns in self.original_vulnerability_data.items():
                if isinstance(package_vulns, list):
                    for vuln in package_vulns:
                        if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                            os_issues.append({
                                'package': package_name,
                                'vulnerability_id': vuln['vulnerability_id'],
                                'description': vuln.get('description', f'Vulnerability {vuln["vulnerability_id"]}'),
                                'severity': vuln.get('severity', vuln.get('cvss_v3_severity', 'UNKNOWN')),
                                'original_vulnerability_data': vuln  # Store complete original data
                            })
                            self.logger.info(f"📋 Using original data for: {vuln['vulnerability_id']} in {package_name}")
        else:
            # Fallback to preprocessing if no stored data
            try:
                processed_logs = self.preprocess_logs_for_ai(logs)
                
                # Try to extract OS scan data from preprocessed logs
                os_scan_pattern = r'=== OS SCAN VULNERABILITIES.*?JSON Format: (\{.*?\})\s*=== END OS SCAN VULNERABILITIES ==='
                os_match = re.search(os_scan_pattern, processed_logs, re.DOTALL)
                
                if os_match:
                    json_str = os_match.group(1)
                    try:
                        vuln_data = json.loads(json_str)
                        
                        for package_name, package_vulns in vuln_data.items():
                            if isinstance(package_vulns, list):
                                for vuln in package_vulns:
                                    if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                                        os_issues.append({
                                            'package': package_name,
                                            'vulnerability_id': vuln['vulnerability_id'],
                                            'description': vuln.get('description', f'Vulnerability {vuln["vulnerability_id"]}'),
                                            'severity': vuln.get('severity', vuln.get('cvss_v3_severity', 'UNKNOWN')),
                                            'original_vulnerability_data': vuln  # Store complete original data
                                        })
                                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"⚠️ Could not parse preprocessed OS scan JSON: {e}")
            
            except Exception as e:
                self.logger.warning(f"⚠️ Could not use preprocessed data: {e}")
        
        # Handle Python vulnerabilities (existing logic)
        safety_pattern = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'(?:.*?advisory=\'([^\']*?)(?:\'|\.\.\.|$))?'
        safety_matches = list(re.finditer(safety_pattern, logs, re.DOTALL))
        
        seen_vulnerabilities = set()
        for match in safety_matches:
            groups = match.groups()
            package = groups[0].strip()
            vuln_id = groups[1].strip()
            advisory = groups[2].strip() if groups[2] else f'Security vulnerability in {package}'
            
            unique_key = (package, vuln_id)
            if unique_key not in seen_vulnerabilities:
                seen_vulnerabilities.add(unique_key)
                py_issues.append({
                    'package': package,
                    'vulnerability_id': vuln_id,
                    'description': advisory,
                    'severity': 'UNKNOWN'
                })
        
        self.logger.info(f"📊 Enhanced fallback found: {len(os_issues)} OS issues, {len(py_issues)} Python issues")
        
        # Create allowlist fixes using original data
        os_allowlist_fixes = []
        for issue in os_issues:
            os_allowlist_fixes.append({
                'vulnerability_id': issue['vulnerability_id'],
                'package': issue['package'],
                'description': issue['description'],
                'original_vulnerability_data': issue.get('original_vulnerability_data')  # Pass through original data
            })
        
        py_allowlist_fixes = []
        for issue in py_issues:
            py_allowlist_fixes.append({
                'vulnerability_id': issue['vulnerability_id'],
                'description': issue['description']
            })
        
        return {
            'os_scan_issues': os_issues,
            'py_scan_issues': py_issues,
            'dockerfile_fixes': [],
            'os_scan_allowlist_fixes': os_allowlist_fixes,
            'py_scan_allowlist_fixes': py_allowlist_fixes,
            'try_dockerfile_first': False,
            'severity_assessment': f'Enhanced fallback found {len(os_issues)} OS and {len(py_issues)} Python vulnerabilities with original data preserved'
        }
    def fallback_rule_based_analysis(self, logs: str) -> Dict:
        """Enhanced fallback to rule-based parsing if AI fails - FIXED duplicates and OS scan parsing"""
        self.logger.info("🔄 Using enhanced fallback rule-based analysis...")
        
        os_issues = []
        py_issues = []
        
        # Parse OS scan JSON format - IMPROVED REGEX
        self.logger.info("🔍 Attempting to parse OS scan vulnerabilities...")
        
        # Try multiple patterns for OS scan data
        os_patterns = [
            # Pattern 1: Standard ECR scan format
            r'Total of \d+ vulnerabilities need to be fixed on [^:]+:\s*(\{.*?\})\s*(?=\n\n|\n[A-Z]|\nINFO|\nERROR|\nWARN|\nE\s+|\n\s*$|$)',
            # Pattern 2: Look for JSON blocks that contain vulnerability_id
            r'(\{[^{}]*"vulnerability_id"[^{}]*\})',
            # Pattern 3: Multi-line JSON with vulnerability data
            r'(\{(?:[^{}]|{[^{}]*})*"vulnerability_id"(?:[^{}]|{[^{}]*})*\})'
        ]
        
        for i, pattern in enumerate(os_patterns):
            self.logger.info(f"🔍 Trying OS scan pattern {i+1}...")
            os_matches = re.finditer(pattern, logs, re.DOTALL | re.MULTILINE)
            
            found_os_data = False
            for match in os_matches:
                json_str = match.group(1)
                try:
                    # Clean up the JSON string
                    json_str = json_str.strip()
                    if not json_str.startswith('{'):
                        continue
                        
                    vuln_data = json.loads(json_str)
                    self.logger.info(f"✅ Successfully parsed OS scan JSON with pattern {i+1}")
                    found_os_data = True
                    
                    # Extract vulnerabilities from each package
                    if isinstance(vuln_data, dict):
                        for package_name, package_vulns in vuln_data.items():
                            if isinstance(package_vulns, list):
                                for vuln in package_vulns:
                                    if isinstance(vuln, dict) and 'vulnerability_id' in vuln:
                                        vuln_id = vuln['vulnerability_id']
                                        
                                        # Check for duplicates
                                        existing = any(
                                            issue['vulnerability_id'] == vuln_id and issue['package'] == package_name 
                                            for issue in os_issues
                                        )
                                        
                                        if not existing:
                                            os_issues.append({
                                                'package': package_name,
                                                'vulnerability_id': vuln_id,
                                                'description': vuln.get('description', f'Vulnerability {vuln_id}'),
                                                'severity': vuln.get('severity', vuln.get('cvss_v3_severity', 'UNKNOWN'))
                                            })
                                            self.logger.info(f"📋 Found OS vulnerability: {vuln_id} in {package_name}")
                                        else:
                                            self.logger.info(f"⚠️ Skipping duplicate OS vulnerability: {vuln_id} in {package_name}")
                                            
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ Could not parse OS scan JSON with pattern {i+1}: {e}")
                    continue
            
            if found_os_data:
                break
        
        # Parse Python scan SAFETY_REPORT format - IMPROVED WITH DUPLICATE DETECTION
        self.logger.info("🔍 Attempting to parse Python scan vulnerabilities...")
        
        # Enhanced pattern to catch various SAFETY_REPORT formats
        safety_patterns = [
            # Pattern 1: Standard format with advisory
            r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'.*?advisory=\'([^\']*?)\'',
            # Pattern 2: Format without advisory or truncated
            r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'',
            # Pattern 3: Just find pkg and vulnerability_id anywhere in the line
            r'\[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\''
        ]
        
        py_vulnerabilities_set = set()  # Use set to avoid duplicates
        
        for i, pattern in enumerate(safety_patterns):
            self.logger.info(f"🔍 Trying PyScan pattern {i+1}...")
            safety_matches = re.finditer(pattern, logs, re.DOTALL)
            
            match_count = 0
            for match in safety_matches:
                match_count += 1
                groups = match.groups()
                
                if len(groups) >= 2:
                    package = groups[0].strip()
                    vuln_id = groups[1].strip()
                    advisory = groups[2].strip() if len(groups) > 2 and groups[2] else f'Security vulnerability in {package}'
                    
                    # Create unique identifier to avoid duplicates
                    unique_key = (package, vuln_id)
                    
                    if unique_key not in py_vulnerabilities_set:
                        py_vulnerabilities_set.add(unique_key)
                        py_issues.append({
                            'package': package,
                            'vulnerability_id': vuln_id,
                            'description': advisory,
                            'severity': 'UNKNOWN'
                        })
                        self.logger.info(f"📋 Found Python vulnerability: {vuln_id} in {package}")
                    else:
                        self.logger.info(f"⚠️ Skipping duplicate Python vulnerability: {vuln_id} in {package}")
            
            if match_count > 0:
                self.logger.info(f"✅ PyScan pattern {i+1} found {match_count} total matches")
                break
        
        self.logger.info(f"📊 Fallback analysis found: {len(os_issues)} unique OS issues, {len(py_issues)} unique Python issues")
        
        # Create allowlist fixes for found issues - NO DUPLICATES
        os_allowlist_fixes = []
        for issue in os_issues:
            os_allowlist_fixes.append({
                'vulnerability_id': issue['vulnerability_id'],
                'package': issue['package'],
                'description': issue['description']
            })
        
        py_allowlist_fixes = []
        for issue in py_issues:
            py_allowlist_fixes.append({
                'vulnerability_id': issue['vulnerability_id'],
                'description': issue['description']
            })
        
        return {
            'os_scan_issues': os_issues,
            'py_scan_issues': py_issues,
            'dockerfile_fixes': [],
            'os_scan_allowlist_fixes': os_allowlist_fixes,
            'py_scan_allowlist_fixes': py_allowlist_fixes,
            'try_dockerfile_first': False,
            'severity_assessment': f'Fallback analysis found {len(os_issues)} unique OS and {len(py_issues)} unique Python vulnerabilities - manual review recommended'
        }

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
            
            # STEP 1: Use AI to extract all vulnerabilities by container type
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
            
            # STEP 2: Try Dockerfile fixes first (container-specific)
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
                # STEP 3: Commit and test the Dockerfile changes
                commit_msg = f"AutoGluon {self.current_version}: Container-specific Dockerfile security fixes"
                if self.commit_and_push_changes(commit_msg):
                    self.logger.info("✅ Container-specific Dockerfile fixes committed, waiting for test results...")
                    
                    # Wait for tests and check results
                    if self.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=40):
                        # STEP 4: Check what's still failing after Dockerfile attempt
                        remaining_failures = self.get_failing_security_tests(pr_number)
                        
                        if not remaining_failures:
                            self.logger.info("🎉 All vulnerabilities fixed with container-specific Dockerfile changes!")
                            return True
                        else:
                            self.logger.info(f"⚠️ {len(remaining_failures)} tests still failing after container-specific Dockerfile fixes")
                            
                            # STEP 5: Handle remaining failures - revert failed fixes and allowlist (container-specific)
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
        """Truncate logs to fit within AI token limits while preserving COMPLETE vulnerability data"""
        if len(logs) <= max_chars:
            return logs
        
        self.logger.info(f"📏 Truncating logs from {len(logs)} to ~{max_chars} chars for AI analysis")
        
        # Step 1: Extract ALL complete vulnerability JSON blocks first (highest priority)
        vulnerability_blocks = []
        lines = logs.split('\n')
        
        # Find all "Total of X vulnerabilities" lines and extract complete JSON blocks
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'Total of' in line and 'vulnerabilities need to be fixed' in line and ':' in line:
                self.logger.info(f"🔍 Found vulnerability line at {i}: {line[:100]}...")
                
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
                    
                    if brace_count == 0:  # Complete JSON found
                        # Include the header line + complete JSON block
                        complete_block = [line] + json_lines
                        block_text = '\n'.join(complete_block)
                        vulnerability_blocks.append(block_text)
                        
                        self.logger.info(f"✅ Extracted complete vulnerability block ({len(block_text)} chars)")
                        
                        # Skip past this block
                        i = json_end_idx + 1
                        continue
            
            i += 1
        
        # Step 2: Extract Python scan vulnerability patterns
        py_vulnerability_lines = []
        for line in lines:
            if ('SAFETY_REPORT' in line and 'FAILED' in line and 'vulnerability_id' in line):
                py_vulnerability_lines.append(line)
        
        # Step 3: Calculate space used by critical vulnerability data
        critical_content = '\n'.join(vulnerability_blocks)
        if py_vulnerability_lines:
            critical_content += '\n' + '\n'.join(py_vulnerability_lines)
        
        critical_size = len(critical_content)
        remaining_space = max_chars - critical_size
        
        self.logger.info(f"📊 Critical vulnerability data: {critical_size} chars")
        self.logger.info(f"📊 Remaining space for other content: {remaining_space} chars")
        
        if critical_size > max_chars:
            self.logger.warning(f"⚠️ Critical vulnerability data ({critical_size} chars) exceeds max_chars ({max_chars})!")
            self.logger.warning("⚠️ Increasing limit to preserve all vulnerability data")
            # Return all critical content even if it exceeds limit - vulnerability data is too important to lose
            return critical_content
        
        # Step 4: Fill remaining space with other important log content
        other_important_lines = []
        
        if remaining_space > 1000:  # Only add other content if we have reasonable space
            for line in lines:
                # Skip lines already included in vulnerability blocks
                skip_line = False
                for block in vulnerability_blocks:
                    if line.strip() in block:
                        skip_line = True
                        break
                
                if not skip_line and line.strip():
                    # Priority content (errors, test results, etc.)
                    if any(keyword in line.lower() for keyword in [
                        'error', 'failed', 'exception', 'traceback', 
                        'test', 'assert', 'cve', 'vulnerability'
                    ]):
                        other_important_lines.append(line)
        
        # Step 5: Combine critical content with other content within limits
        final_content = critical_content
        
        if other_important_lines and remaining_space > 100:
            other_content = '\n'.join(other_important_lines)
            if len(other_content) <= remaining_space:
                final_content += '\n\n=== OTHER LOG CONTENT ===\n' + other_content
            else:
                # Truncate other content to fit
                truncated_other = other_content[:remaining_space-50] + '\n[... truncated ...]'
                final_content += '\n\n=== OTHER LOG CONTENT ===\n' + truncated_other
        
        final_size = len(final_content)
        self.logger.info(f"📏 Final truncated size: {final_size} chars (preserved {len(vulnerability_blocks)} vulnerability blocks)")
        
        # Verify all vulnerability blocks are intact
        for i, block in enumerate(vulnerability_blocks):
            if block in final_content:
                self.logger.info(f"✅ Vulnerability block {i+1} preserved intact")
            else:
                self.logger.error(f"❌ Vulnerability block {i+1} was corrupted during truncation!")
        
        return final_content
    
    def detect_container_type_from_test_name(self, test_name: str) -> str:
        """Detect container type from test name"""
        test_name_lower = test_name.lower()
        
        if 'training' in test_name_lower:
            return 'training'
        elif 'inference' in test_name_lower:
            return 'inference'
        else:
            # Try to detect from other patterns
            if any(pattern in test_name_lower for pattern in ['train', 'training-']):
                return 'training'
            elif any(pattern in test_name_lower for pattern in ['infer', 'inference-']):
                return 'inference'
            else:
                return 'unknown'
    ## 2. New `ai_detect_vulnerabilities_only()` method:

    def ai_detect_vulnerabilities_only(self, logs: str) -> Dict:
        """Use AI only for vulnerability detection - no analysis or formatting"""
        self.logger.info("🤖 Using AI only for vulnerability detection...")
        
        try:
            # Truncate logs for AI processing
            truncated_logs = self.truncate_logs_for_ai(logs, max_chars=50000)
            
            # Preprocess logs to help AI detection
            processed_logs = self.preprocess_logs_for_ai(truncated_logs)
            
            self.logger.info("🧠 AI is detecting vulnerabilities...")
            ai_response = self.detection_chain.invoke({"security_logs": processed_logs})
            
            self.logger.info("="*80)
            self.logger.info("🔍 DEBUG: AI DETECTION RESPONSE")
            self.logger.info("="*80)
            self.logger.info(f"Response type: {type(ai_response)}")
            self.logger.info(f"Response content: {ai_response}")
            self.logger.info("="*80)
            
            # Parse AI response
            if isinstance(ai_response, dict):
                detected = ai_response
            else:
                # Try to parse if it's a string
                try:
                    detected = json.loads(str(ai_response))
                except json.JSONDecodeError:
                    self.logger.error("❌ AI returned invalid JSON for detection")
                    return {'os_vulns': [], 'py_vulns': []}
            
            os_vulns = detected.get('os_vulns', [])
            py_vulns = detected.get('py_vulns', [])
            
            self.logger.info(f"🤖 AI detected {len(os_vulns)} OS and {len(py_vulns)} Python vulnerabilities")
            
            return {
                'os_vulns': os_vulns,
                'py_vulns': py_vulns
            }
            
        except Exception as e:
            self.logger.error(f"❌ AI detection failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'os_vulns': [], 'py_vulns': []}
        
    def extract_all_vulnerabilities(self, logs: str) -> Dict:
        """Extract all vulnerabilities using AI for detection only, then rule-based formatting"""
        self.logger.info("🤖 Using AI only for vulnerability detection, rule-based for formatting...")
        
        try:
            # Step 1: Use AI only to detect which vulnerabilities exist
            detected_vulnerabilities = self.ai_detect_vulnerabilities_only(logs)
            
            # Step 2: Use rule-based logic to get full vulnerability data from preprocessed logs
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
            
            # Step 3: Match detected vulnerabilities with original data and format properly
            for detected_vuln in detected_vulnerabilities.get('os_vulns', []):
                package = detected_vuln.get('package', '').lower()
                vuln_id = detected_vuln.get('vulnerability_id', '')
                
                # Find original vulnerability data
                original_data = self.find_original_vulnerability_data(package, vuln_id)
                
                if original_data:
                    os_vulnerabilities.append({
                        'package': package,
                        'vulnerability_id': vuln_id,
                        'description': original_data.get('description', f'Vulnerability {vuln_id}'),
                        'severity': original_data.get('severity', original_data.get('cvss_v3_severity', 'UNKNOWN')),
                        'original_data': original_data
                    })
                    self.logger.info(f"✅ Matched OS vulnerability with original data: {vuln_id} in {package}")
                else:
                    # Fallback if no original data found
                    os_vulnerabilities.append({
                        'package': package,
                        'vulnerability_id': vuln_id,
                        'description': detected_vuln.get('description', f'Vulnerability {vuln_id}'),
                        'severity': detected_vuln.get('severity', 'UNKNOWN'),
                        'original_data': {}
                    })
                    self.logger.warning(f"⚠️ No original data found for OS vulnerability: {vuln_id} in {package}")
            
            # Handle Python vulnerabilities (simpler structure)
            for detected_vuln in detected_vulnerabilities.get('py_vulns', []):
                py_vulnerabilities.append({
                    'package': detected_vuln.get('package', ''),
                    'vulnerability_id': detected_vuln.get('vulnerability_id', ''),
                    'description': detected_vuln.get('description', ''),
                    'severity': detected_vuln.get('severity', 'UNKNOWN')
                })
                self.logger.info(f"✅ Added Python vulnerability: {detected_vuln.get('vulnerability_id')} in {detected_vuln.get('package')}")
            
            self.logger.info(f"📊 Final result: {len(os_vulnerabilities)} OS and {len(py_vulnerabilities)} Python vulnerabilities with proper formatting")
            
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
                                self.logger.info(f"📋 Found original data for {vulnerability_id} in {package_name}")
                                
                                # Print the original data structure for debugging
                                self.logger.info("="*80)
                                self.logger.info(f"🔍 ORIGINAL VULNERABILITY DATA FOR {vulnerability_id}")
                                self.logger.info("="*80)
                                self.logger.info(json.dumps(vuln, indent=2))
                                self.logger.info("="*80)
                                
                                return vuln
                
                # If exact package name not found, try variations
                for stored_package, stored_vulns in self.original_vulnerability_data.items():
                    if isinstance(stored_vulns, list):
                        for vuln in stored_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                # Check if package names are similar
                                if (package_name.lower() in stored_package.lower() or 
                                    stored_package.lower() in package_name.lower()):
                                    self.logger.info(f"📋 Found original data for {vulnerability_id} via package match: {stored_package}")
                                    return vuln
            
            self.logger.warning(f"⚠️ No original vulnerability data found for {vulnerability_id} in {package_name}")
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error finding original vulnerability data: {e}")
            return {}
    def extract_all_vulnerabilities_by_container(self, all_logs: str) -> Dict:
        """Extract vulnerabilities organized by container type"""
        self.logger.info("🤖 Extracting vulnerabilities by container type...")
        
        # Initialize result structure
        result = {
            'training': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'inference': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'unknown': {'os_vulnerabilities': [], 'py_vulnerabilities': []}
        }
        
        # Process each container type's logs separately
        for container_type, logs in self.container_specific_logs.items():
            if not logs.strip():
                self.logger.info(f"📊 No logs for {container_type} container")
                continue
                
            self.logger.info(f"📊 Processing {container_type} container logs ({len(logs)} chars)")
            
            # Extract vulnerabilities for this container type
            try:
                container_vulns = self.extract_all_vulnerabilities(logs)
                
                # Add container type to each vulnerability
                for vuln in container_vulns['os_vulnerabilities']:
                    vuln['container_type'] = container_type
                    result[container_type]['os_vulnerabilities'].append(vuln)
                    
                for vuln in container_vulns['py_vulnerabilities']:
                    vuln['container_type'] = container_type
                    result[container_type]['py_vulnerabilities'].append(vuln)
                    
                self.logger.info(f"📊 {container_type}: {len(container_vulns['os_vulnerabilities'])} OS, {len(container_vulns['py_vulnerabilities'])} Python vulnerabilities")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to extract vulnerabilities for {container_type}: {e}")
        
        # Store in instance variable for later use
        self.container_specific_vulnerabilities = result
        
        # Return the container-specific structure (not combined)
        return result


    def extract_all_vulnerabilities_fallback(self, logs: str) -> Dict:
        """Fallback rule-based extraction when AI fails"""
        self.logger.info("🔍 Using fallback rule-based vulnerability extraction...")
        
        # Use preprocessing to extract vulnerability data
        processed_logs = self.preprocess_logs_for_ai(logs)
        
        os_vulnerabilities = []
        py_vulnerabilities = []
        
        # Extract OS vulnerabilities from stored original data
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
        
        # Extract Python vulnerabilities  
        safety_pattern = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'(?:.*?advisory=\'([^\']*?)(?:\'|\.\.\.|$))?'
        safety_matches = list(re.finditer(safety_pattern, logs, re.DOTALL))
        
        seen_py_vulns = set()
        for match in safety_matches:
            groups = match.groups()
            package = groups[0].strip()
            vuln_id = groups[1].strip()
            advisory = groups[2].strip() if groups[2] else f'Security vulnerability in {package}'
            
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
    def get_available_versions_pip_index(self, package_name: str) -> List[str]:
        """Get available versions using pip index versions command"""
        try:
            import subprocess
            
            self.logger.info(f"🔍 Checking available versions for {package_name} via pip index")
            
            # Run pip index versions command
            result = subprocess.run(
                ["pip", "index", "versions", package_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.logger.warning(f"⚠️ pip index failed for {package_name}: {result.stderr}")
                return []
            
            # Parse output - typically shows "Available versions: 3.9.0, 3.8.2, 3.8.1, ..."
            output = result.stdout.strip()
            versions = []
            
            # Look for "Available versions:" line
            for line in output.split('\n'):
                if 'Available versions:' in line:
                    # Extract versions after the colon
                    version_part = line.split('Available versions:')[1].strip()
                    # Split by comma and clean up
                    raw_versions = [v.strip() for v in version_part.split(',')]
                    
                    # Filter to stable versions only
                    for version in raw_versions:
                        # Remove any whitespace and check format
                        version = version.strip()
                        if re.match(r'^\d+\.\d+(\.\d+)?$', version):
                            versions.append(version)
                    
                    break
            
            self.logger.info(f"📦 Found {len(versions)} available versions for {package_name}")
            if versions:
                self.logger.info(f"   Latest versions: {versions[:5]}")
            
            return versions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get pip index versions for {package_name}: {e}")
            return []

    def check_upgrade_possible_in_constraint(self, package_name: str, constraint: str, current_version_from_logs: str) -> bool:
        """Check if upgrade is possible within constraint range"""
        try:
            self.logger.info(f"🔍 Checking if upgrade possible for {package_name}: {current_version_from_logs} → {constraint}")
            
            # Get available versions
            available_versions = self.get_available_versions_pip_index(package_name)
            if not available_versions:
                self.logger.warning(f"⚠️ No available versions found for {package_name}")
                return False
            
            # Filter versions that satisfy the constraint
            valid_versions = self.filter_versions_by_constraint(available_versions, constraint)
            
            if not valid_versions:
                self.logger.warning(f"❌ No versions satisfy constraint '{constraint}' for {package_name}")
                return False
            
            # Check if any valid version is newer than current
            current_parts = [int(x) for x in current_version_from_logs.split('.')]
            
            for version in valid_versions:
                try:
                    version_parts = [int(x) for x in version.split('.')]
                    
                    # Pad shorter version with zeros for comparison
                    max_len = max(len(current_parts), len(version_parts))
                    current_padded = current_parts + [0] * (max_len - len(current_parts))
                    version_padded = version_parts + [0] * (max_len - len(version_parts))
                    
                    # If this version is newer than current, upgrade is possible
                    if version_padded > current_padded:
                        self.logger.info(f"✅ Upgrade possible: {current_version_from_logs} → {version} (within constraint {constraint})")
                        return True
                        
                except (ValueError, IndexError):
                    continue
            
            self.logger.warning(f"❌ No newer versions found within constraint '{constraint}' for {package_name}")
            self.logger.info(f"   Current: {current_version_from_logs}")
            self.logger.info(f"   Valid versions in constraint: {valid_versions[:3]}")
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check upgrade possibility for {package_name}: {e}")
            return False

    def get_optimal_version_constraint_with_upgrade_check(self, package_name: str, vuln_data: Dict, all_logs: str, vulnerability_type: str = 'os_scan') -> str:
        """Get optimal constraint and check if upgrade is actually possible - ENHANCED WITH VULNERABILITY TYPE"""
        # Get the constraint using vulnerability-type-specific logic
        constraint = self.get_optimal_version_constraint(package_name, vuln_data, all_logs, vulnerability_type)
        
        if constraint == "latest":
            return constraint
        
        # Get current version from logs or vulnerability data
        current_version_from_logs = self.extract_current_version_from_logs(package_name, all_logs)
        
        if not current_version_from_logs:
            # Try to get current version from vulnerability data
            current_version_from_logs = vuln_data.get('package_details', {}).get('version', '')
            if current_version_from_logs:
                # Clean version (remove stuff like "+cpu")
                current_version_from_logs = re.sub(r'\+.*$', '', current_version_from_logs)
                self.logger.info(f"📋 Using vulnerability data version for {package_name}: {current_version_from_logs}")
        
        if not current_version_from_logs:
            self.logger.warning(f"⚠️ Could not extract current version for {package_name}, trying constraint anyway")
            return constraint
        
        # Check if upgrade is possible within this constraint
        if self.check_upgrade_possible_in_constraint(package_name, constraint, current_version_from_logs):
            self.logger.info(f"✅ Upgrade possible for {package_name} with constraint '{constraint}' ({vulnerability_type})")
            return constraint
        else:
            self.logger.warning(f"❌ No upgrade possible for {package_name} with constraint '{constraint}' ({vulnerability_type}) - will skip Dockerfile")
            return "skip_dockerfile"  # Special marker to skip Dockerfile attempts

    def check_pyscan_constraint_and_cross_contaminate(self, vulnerabilities: Dict, all_logs: str) -> Dict:
        """Check pyscan constraints and cross-contaminate to OS scan if needed"""
        self.logger.info("🔍 Checking pyscan constraints for cross-contamination...")
        
        failed_pyscan_packages = set()
        
        # Check all pyscan vulnerabilities first
        for vuln in vulnerabilities['py_vulnerabilities']:
            package = vuln['package']
            
            # Get pyscan constraint
            pyscan_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                package, {}, all_logs, 'py_scan'
            )
            
            if pyscan_constraint == "skip_dockerfile":
                self.logger.warning(f"⚠️ Pyscan constraint failed for {package} - will cross-contaminate to OS scan")
                failed_pyscan_packages.add(package)
        
        # Cross-contaminate: if pyscan failed, also allowlist any OS scan vulnerabilities for the same package
        if failed_pyscan_packages:
            self.logger.info(f"🔄 Cross-contaminating {len(failed_pyscan_packages)} packages: {failed_pyscan_packages}")
            
            for vuln in vulnerabilities['os_vulnerabilities']:
                if vuln['package'] in failed_pyscan_packages:
                    vuln['cross_contaminated'] = True
                    self.logger.info(f"🔄 Marked OS scan vulnerability {vuln['vulnerability_id']} for {vuln['package']} as cross-contaminated")
        
        return vulnerabilities

    def attempt_dockerfile_fixes_first_by_container(self, vulnerabilities: Dict) -> bool:
        """Apply Dockerfile fixes only to relevant container types - ENHANCED WITH PYSCAN CROSS-CONTAMINATION"""
        self.logger.info("🔧 Applying container-specific Dockerfile fixes with pyscan cross-contamination...")
        
        # First, check pyscan constraints and cross-contaminate if needed
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
            
            # Normal OS scan processing (description-based only)
            if self.should_skip_dockerfile_fix(package, 'os_scan'):
                container_fixes[container_type]['os_allowlist'].append({
                    'vulnerability_id': vuln_id,
                    'package': package,
                    'description': vuln['description'],
                    'original_vulnerability_data': vuln.get('original_data')
                })
                self.logger.info(f"📝 Pre-filtered to {container_type} OS allowlist: {vuln_id} in {package}")
            else:
                # Get optimal version constraint for OS scan (description parsing only)
                version_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                    package, 
                    vuln.get('original_data', {}), 
                    all_logs,
                    'os_scan'  # OS scan uses description parsing only
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
                # Get optimal version constraint for pyscan (Safety spec only)
                version_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                    package, 
                    {}, 
                    all_logs,
                    'py_scan'  # Python scan uses Safety spec only
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
                
                if container_type != 'unknown':  # Only apply to known container types
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
        """Handle vulnerabilities that persist after Dockerfile fixes by reverting and allowlisting (container-specific)"""
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
        
        # Check if any persistent vulnerabilities found
        total_persistent = sum(len(container_vulns['os_vulnerabilities']) + len(container_vulns['py_vulnerabilities']) 
                            for container_vulns in container_persistent_vulns.values())
        
        if total_persistent == 0:
            self.logger.info("✅ No persistent vulnerabilities detected - Dockerfile fixes worked for attempted packages")
            return True
        
        # Revert the Dockerfile changes for packages that didn't work (by container type)
        for container_type, packages_to_revert in packages_to_revert_by_container.items():
            if packages_to_revert and container_type != 'unknown':
                self.logger.info(f"🔄 Reverting {container_type} Dockerfile fixes for {len(packages_to_revert)} packages that didn't work")
                revert_success = self.selectively_revert_dockerfile_packages(container_type, packages_to_revert)
                if not revert_success:
                    self.logger.error(f"❌ Failed to revert packages for {container_type}")
        
        # Apply allowlist fixes by container type
        allowlist_success = True
        for container_type, persistent_vulns in container_persistent_vulns.items():
            if container_type == 'unknown':
                continue
                
            os_vulns = persistent_vulns['os_vulnerabilities']
            py_vulns = persistent_vulns['py_vulnerabilities']
            
            if not os_vulns and not py_vulns:
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
            
            # Apply allowlist fixes to this specific container type
            self.logger.info(f"📝 Applying {len(os_allowlist_fixes)} OS and {len(py_allowlist_fixes)} Python allowlist fixes to {container_type}")
            container_success = self.apply_allowlist_fixes(container_type, os_allowlist_fixes, py_allowlist_fixes)
            if not container_success:
                allowlist_success = False
        
        if allowlist_success:
            # Commit the revert + allowlist changes
            commit_msg = f"AutoGluon {self.current_version}: Revert failed container-specific fixes and apply additional allowlists"
            if self.commit_and_push_changes(commit_msg):
                self.logger.info("✅ Reverted failed fixes and applied additional container-specific allowlists")
                
                # Wait for final test results
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

    def handle_remaining_vulnerabilities_with_allowlist(self, pr_number: int, original_vulnerabilities: Dict, remaining_failures: List[Dict]) -> bool:
        """Handle vulnerabilities that persist after Dockerfile fixes by reverting and allowlisting"""
        self.logger.info("🔄 Handling remaining vulnerabilities - reverting failed fixes and allowlisting...")
        
        # Get logs for remaining failures to see which vulnerabilities persist
        remaining_logs = self.collect_security_logs(remaining_failures)
        remaining_vulns = self.extract_all_vulnerabilities(remaining_logs)
        
        # Determine which vulnerabilities still exist - ONLY CHECK DOCKERFILE ATTEMPTED ONES
        persistent_os_vulns = []
        persistent_py_vulns = []
        
        # Find vulnerabilities that are still present (only ones we tried to fix with Dockerfile)
        for vuln in original_vulnerabilities['os_vulnerabilities']:
            # Only check vulnerabilities that we tried to fix with Dockerfile (not pre-filtered ones)
            was_dockerfile_attempt = any(
                fix['vulnerability_id'] == vuln['vulnerability_id'] 
                for fix in getattr(self, 'attempted_dockerfile_fixes', [])
            )
            
            if was_dockerfile_attempt:
                still_exists = any(
                    rv['vulnerability_id'] == vuln['vulnerability_id'] and rv['package'] == vuln['package']
                    for rv in remaining_vulns['os_vulnerabilities']
                )
                if still_exists:
                    persistent_os_vulns.append(vuln)
                    self.logger.info(f"🔄 OS vulnerability persists after Dockerfile fix: {vuln['vulnerability_id']} in {vuln['package']}")
        
        for vuln in original_vulnerabilities['py_vulnerabilities']:
            was_dockerfile_attempt = any(
                fix['vulnerability_id'] == vuln['vulnerability_id'] 
                for fix in getattr(self, 'attempted_dockerfile_fixes', [])
            )
            
            if was_dockerfile_attempt:
                still_exists = any(
                    rv['vulnerability_id'] == vuln['vulnerability_id'] and rv['package'] == vuln['package']  
                    for rv in remaining_vulns['py_vulnerabilities']
                )
                if still_exists:
                    persistent_py_vulns.append(vuln)
                    self.logger.info(f"🔄 Python vulnerability persists after Dockerfile fix: {vuln['vulnerability_id']} in {vuln['package']}")
        
        if not persistent_os_vulns and not persistent_py_vulns:
            self.logger.info("✅ No persistent vulnerabilities detected - Dockerfile fixes worked for attempted packages")
            return True
        
        # Revert the Dockerfile changes for packages that didn't work
        packages_to_revert = []
        for vuln in persistent_os_vulns + persistent_py_vulns:
            packages_to_revert.append(vuln['package'])
        
        if packages_to_revert:
            self.logger.info(f"🔄 Reverting Dockerfile fixes for {len(packages_to_revert)} packages that didn't work")
            for container_type in ['training', 'inference']:
                revert_success = self.selectively_revert_dockerfile_packages(container_type, packages_to_revert)
                if not revert_success:
                    self.logger.error(f"❌ Failed to revert packages for {container_type}")
        
        # Now add persistent vulnerabilities to allowlist
        os_allowlist_fixes = []
        for vuln in persistent_os_vulns:
            os_allowlist_fixes.append({
                'vulnerability_id': vuln['vulnerability_id'],
                'package': vuln['package'],
                'description': vuln['description'],
                'original_vulnerability_data': vuln.get('original_data')
            })
        
        py_allowlist_fixes = []
        for vuln in persistent_py_vulns:
            py_allowlist_fixes.append({
                'vulnerability_id': vuln['vulnerability_id'],
                'description': vuln['description']
            })
        
        # Apply allowlist fixes
        allowlist_success = True
        for container_type in ['training', 'inference']:
            container_success = self.apply_allowlist_fixes(container_type, os_allowlist_fixes, py_allowlist_fixes)
            if not container_success:
                allowlist_success = False
        
        if allowlist_success:
            # Commit the revert + allowlist changes
            commit_msg = f"AutoGluon {self.current_version}: Revert failed Dockerfile fixes and apply additional allowlist"
            if self.commit_and_push_changes(commit_msg):
                self.logger.info("✅ Reverted failed fixes and applied additional allowlist")
                
                # Wait for final test results
                if self.wait_for_security_tests_to_complete(pr_number, max_wait_minutes=20):
                    final_failures = self.get_failing_security_tests(pr_number)
                    if not final_failures:
                        self.logger.info("🎉 All vulnerabilities resolved!")
                        return True
                    else:
                        self.logger.warning(f"⚠️ {len(final_failures)} tests still failing after additional allowlist")
                        return False
                else:
                    self.logger.warning("⚠️ Timeout waiting for final test results")
                    return False
            else:
                self.logger.error("❌ Failed to commit allowlist changes")
                return False
        else:
            self.logger.error("❌ Failed to apply allowlist fixes")
            return False
    def apply_direct_allowlist_fixes(self, vulnerabilities: Dict) -> bool:
        """Apply allowlist fixes directly when Dockerfile approach isn't possible"""
        self.logger.info("📝 Applying direct allowlist fixes...")
        
        # Create allowlist fixes
        os_allowlist_fixes = []
        for vuln in vulnerabilities['os_vulnerabilities']:
            os_allowlist_fixes.append({
                'vulnerability_id': vuln['vulnerability_id'],
                'package': vuln['package'],
                'description': vuln['description'],
                'original_vulnerability_data': vuln.get('original_data')
            })
        
        py_allowlist_fixes = []
        for vuln in vulnerabilities['py_vulnerabilities']:
            py_allowlist_fixes.append({
                'vulnerability_id': vuln['vulnerability_id'],
                'description': vuln['description']
            })
        
        # Apply to both container types
        success = True
        for container_type in ['training', 'inference']:
            container_success = self.apply_allowlist_fixes(container_type, os_allowlist_fixes, py_allowlist_fixes)
            if not container_success:
                success = False
        
        if success:
            commit_msg = f"AutoGluon {self.current_version}: Apply security allowlist fixes"
            return self.commit_and_push_changes(commit_msg)
        
        return success
    def parse_safety_spec_to_constraint(self, spec: str, package_name: str, logs: str) -> str:
        """Parse Safety spec into proper pip constraint - CORRECTED LOGIC"""
        try:
            # Get current installed version from logs
            current_version = self.extract_current_version_from_logs(package_name, logs)
            
            self.logger.info(f"📋 Parsing Safety spec '{spec}' for {package_name} (current: {current_version})")
            
            if spec.startswith('<'):
                # spec='<3.9' means safe versions are < 3.9
                max_version = spec[1:].strip()
                
                if current_version:
                    # Option: upgrade from current but stay under max_version
                    # Format: ">current_version,<max_version"
                    constraint = f">{current_version},<{max_version}"
                    self.logger.info(f"📋 Safety spec '<{max_version}' with current {current_version} → {constraint}")
                    return constraint
                else:
                    # Just use the spec directly
                    self.logger.info(f"📋 Safety spec '<{max_version}' → <{max_version}")
                    return f"<{max_version}"
            
            elif spec.startswith('<='):
                # spec='<=3.8' means safe versions are <= 3.8
                max_version = spec[2:].strip()
                
                if current_version:
                    constraint = f">{current_version},<={max_version}"
                    self.logger.info(f"📋 Safety spec '<={max_version}' with current {current_version} → {constraint}")
                    return constraint
                else:
                    return f"<={max_version}"
            
            elif spec.startswith('>='):
                # spec='>=3.9' means safe versions are >= 3.9
                min_version = spec[2:].strip()
                return f">={min_version}"
            
            elif spec.startswith('>'):
                # spec='>3.9' means safe versions are > 3.9
                min_version = spec[1:].strip()
                return f">{min_version}"
            
            elif ',' in spec:
                # Complex spec like ">=1.0,<2.0" - use as-is
                return spec
            
            else:
                # Exact version or other format
                return f"=={spec}"
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not parse Safety spec '{spec}': {e}")
            return "latest"

    def parse_version_constraints_from_logs(self, logs: str, package_name: str) -> str:
        """Parse version constraints from Safety reports and remediation information in logs - UPDATED"""
        try:
            # Look for Safety report spec patterns - PRIORITY SOURCE
            safety_spec_patterns = [
                # Pattern: spec='<3.9' or spec="<3.9"
                rf'\[pkg: {re.escape(package_name)}\].*?spec=[\'"]([^\'\"]+)[\'"]',
                # Pattern: any spec='...' near the package name
                rf'{re.escape(package_name)}.*?spec=[\'"]([^\'\"]+)[\'"]',
                # Pattern: spec='...' anywhere in a line mentioning the package
                rf'spec=[\'"]([^\'\"]+)[\'"].*?{re.escape(package_name)}',
            ]
            
            for pattern in safety_spec_patterns:
                matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if matches:
                    spec = matches[0].strip()
                    self.logger.info(f"📋 Found Safety spec for {package_name}: {spec}")
                    
                    # Parse the spec to get the constraint
                    constraint = self.parse_safety_spec_to_constraint(spec, package_name, logs)
                    if constraint != "latest":
                        return constraint
            
            # Fallback to other patterns if no Safety spec found
            # Look for pip install commands with version constraints
            pip_patterns = [
                rf'pip install[^"]*"({re.escape(package_name)}[>=<,0-9\.\s]+)"',
                rf'pip install[^"]*\s({re.escape(package_name)}[>=<,0-9\.\s]+)(?:\s|$)',
                rf'"({re.escape(package_name)}[>=<,0-9\.\s]+)"',
            ]
            
            for pattern in pip_patterns:
                matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE)
                if matches:
                    full_constraint = matches[0].strip()
                    if full_constraint.startswith(package_name):
                        constraint = full_constraint[len(package_name):].lstrip()
                        self.logger.info(f"📋 Found pip constraint in logs for {package_name}: {constraint}")
                        return constraint
            
            # Look for OS scan "through" patterns in descriptions
            desc_patterns = [
                rf'{re.escape(package_name)} through ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'{re.escape(package_name)}.*?before ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
            ]
            
            for pattern in desc_patterns:
                matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE)
                if matches:
                    version = matches[0]
                    if 'through' in pattern:
                        # "through X" means we need X+1 or higher
                        constraint = self.increment_version(version)
                        self.logger.info(f"📋 Found 'through' version for {package_name}: >={constraint}")
                        return f">={constraint}"
                    elif 'before' in pattern:
                        # "before X" means we need >= X
                        self.logger.info(f"📋 Found 'before' version for {package_name}: >={version}")
                        return f">={version}"
            
            return "latest"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not parse version constraints from logs for {package_name}: {e}")
            return "latest"
        
    def extract_current_version_from_logs(self, package_name: str, logs: str) -> str:
        """Extract current installed version from Safety report logs"""
        try:
            # Look for [installed: X.X.X] pattern
            version_patterns = [
                rf'\[pkg: {re.escape(package_name)}\].*?\[installed: ([^\]]+)\]',
                rf'{re.escape(package_name)}.*?\[installed: ([^\]]+)\]',
            ]
            
            for pattern in version_patterns:
                matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE)
                if matches:
                    version = matches[0].strip()
                    self.logger.info(f"📋 Found current installed version for {package_name}: {version}")
                    return version
            
            return ""
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract current version for {package_name}: {e}")
            return ""

    def increment_version(self, version: str, increment_patch: bool = False) -> str:
        """Increment version number safely"""
        try:
            parts = version.split('.')
            if len(parts) >= 2:
                major = int(parts[0])
                minor = int(parts[1])
                patch = int(parts[2]) if len(parts) > 2 else 0
                
                if increment_patch:
                    # Increment patch version
                    return f"{major}.{minor}.{patch + 1}"
                else:
                    # Increment minor version
                    return f"{major}.{minor + 1}.0"
            else:
                # Just append .1
                return f"{version}.1"
        except (ValueError, IndexError):
            return version
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
    
    if args.debug_tests:
        # Run debug analysis only
        pr_number = args.pr_number or agent.get_current_pr_number()
        if pr_number:
            agent.debug_all_available_tests(pr_number)
        else:
            print("❌ No PR number provided and no current PR found")
        exit(0)
    
    success = agent.run_security_test_agent()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()