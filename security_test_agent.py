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

from common import BaseAutomation

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
        
        # FIXED DETECTION PROMPT - PRESERVE UPGRADE INFORMATION
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
        "description": "description text INCLUDING upgrade instructions",
        "severity": "CRITICAL|HIGH|MEDIUM|LOW"
        }}
    ],
    "py_vulns": [
        {{
        "package": "package_name",
        "vulnerability_id": "vulnerability_id",
        "description": "description text INCLUDING upgrade instructions",
        "severity": "UNKNOWN"
        }}
    ]
    }}

    ## CRITICAL INSTRUCTIONS

    1. Extract ALL vulnerabilities from both OS scan JSON and Python Safety reports
    2. Be precise - only extract what you can clearly identify
    3. Don't make decisions - just detect and extract
    4. Use exact vulnerability IDs and package names from the logs
    5. **PRESERVE UPGRADE INFORMATION**: If the description contains upgrade instructions like "upgrade to version X.X.X" or "Users should upgrade to version X.X.X", include this in the description - do NOT truncate it
    6. **PRESERVE VERSION NUMBERS**: Keep any version numbers mentioned in upgrade instructions
    7. For descriptions, include the main vulnerability text AND any upgrade/remediation guidance

    Focus on accurate detection, not analysis or recommendations. Preserve upgrade information for version extraction."""),
    ("human", """Extract all vulnerabilities from these security logs:
    Security Test Logs:
    {security_logs}
    Extract every vulnerability you can find - both OS scan CVEs and Python Safety reports. Return only the detection results in JSON format. PRESERVE any upgrade instructions and version numbers in descriptions.""")
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
            
            
            self.logger.info(f"\n🔍 GraphQL API (current HEAD only):")
            graphql_tests = self._get_graphql_tests_for_commit(pr_number, head_sha)
            if graphql_tests:
                self.logger.info(f"   Found {len(graphql_tests)} tests via GraphQL for HEAD")
                for test in graphql_tests:
                    test_name = test['name']
                    is_security = self._is_security_test(test_name)
                    
                    
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
                
                
                self.logger.info(f"\n🔍 Checking recent commits for AutoGluon tests...")
                recent_autogluon_tests = self._get_recent_autogluon_tests(pr_number)
                if recent_autogluon_tests:
                    self.logger.info(f"   Found AutoGluon tests in recent commits:")
                    for test in recent_autogluon_tests:
                        security_indicator = "🛡️ SECURITY" if self._is_security_test(test['name']) else "   regular"
                        self.logger.info(f"   {security_indicator}: {test['name']} ({test.get('state', 'UNKNOWN')}) [commit: {test.get('commit_oid', 'unknown')[:8]}]")
                else:
                    self.logger.info(f"   No AutoGluon tests found in recent commits either")
            
            
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
            
            
            if failing_tests:
                self.logger.info(f"\n❌ CURRENTLY FAILING TESTS ({len(failing_tests)}):")
                for test in failing_tests:
                    security_indicator = "🛡️ SECURITY" if test['is_security'] else "   regular"
                    self.logger.info(f"   {security_indicator}: {test['name']}")
            else:
                self.logger.info(f"\n✅ No tests currently failing")
            
            
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
    def extract_full_description_from_logs(self, package_name: str, vulnerability_id: str, logs: str) -> str:
        """Extract full description from logs with multiple fallback patterns"""
        try:
            # Pattern 1: Try to get full advisory content between quotes
            full_advisory_pattern = rf'vulnerability_id=\'{re.escape(vulnerability_id)}\'.*?advisory=\'([^\']+)\''
            match = re.search(full_advisory_pattern, logs, re.DOTALL)
            if match:
                full_description = match.group(1).strip()
                self.logger.info(f"📋 Found full description for {vulnerability_id}: {len(full_description)} chars")
                return full_description
            
            # Pattern 2: Try to get description from JSON vulnerability data
            json_pattern = rf'"{re.escape(package_name)}".*?"description":\s*"([^"]+)"'
            match = re.search(json_pattern, logs, re.DOTALL | re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                self.logger.info(f"📋 Found JSON description for {vulnerability_id}: {len(description)} chars")
                return description
            
            # Pattern 3: Fallback to original short description
            safety_pattern = rf'SAFETY_REPORT.*?\[pkg: {re.escape(package_name)}\].*?vulnerability_id=\'{re.escape(vulnerability_id)}\'.*?advisory=\'([^\']*?)\''
            match = re.search(safety_pattern, logs, re.DOTALL)
            if match:
                description = match.group(1).strip()
                self.logger.info(f"📋 Found basic description for {vulnerability_id}: {len(description)} chars")
                return description
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract description for {vulnerability_id}: {e}")
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
                        if isinstance(match, tuple):
                            match = match[0] if match else ""
                        
                        clean_content = re.sub(r'<[^>]+>', '', str(match))
                        clean_content = clean_content.replace('&lt;', '<').replace('&gt;', '>')
                        clean_content = clean_content.replace('&amp;', '&').replace('&quot;', '"')
                        clean_content = clean_content.replace('\\n', '\n').replace('\\t', '\t')
                        clean_content = clean_content.strip()
                        if len(clean_content) > 50:
                            extracted_content.append(clean_content)
            if extracted_content:
                combined_logs = '\n---\n'.join(extracted_content)
                self.logger.info(f"✅ Extracted logs from CodeBuild HTML ({len(combined_logs)} chars)")
                return combined_logs
            else:
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
                # Apply OS scan allowlist fixes (ECR Enhanced Scan format - using ORIGINAL data)
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
        """Extract specific version constraint from vulnerability remediation data and description - ENHANCED AND STRICT"""
        try:
            # First check remediation recommendation
            remediation = vuln_data.get('remediation', {})
            recommendation = remediation.get('recommendation', {})
            recommendation_text = recommendation.get('text', '')
            
            if recommendation_text and recommendation_text != "None Provided":
                # Look for version patterns in recommendation text (strict patterns only)
                version_patterns = [
                    # Pattern: "upgrade to version X.X.X or later"
                    r'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)(?: or later)?',
                    # Pattern: "fixed in version X.X.X"
                    r'fixed in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                    # Pattern: "version >= X.X.X"
                    r'version >= ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                ]
                
                for pattern in version_patterns:
                    match = re.search(pattern, recommendation_text, re.IGNORECASE)
                    if match:
                        fixed_version = match.group(1)
                        self.logger.info(f"📋 Found fixed version in remediation for {package_name}: {fixed_version}")
                        return f">={fixed_version}"
            
            # Check description field for version information - STRICT PATTERNS ONLY
            description = vuln_data.get('description', '')
            if description:
                self.logger.info(f"🔍 Checking description for {package_name}: {description[:100]}...")
                description_constraint = self.extract_version_from_description_patterns(description, package_name)
                if description_constraint and description_constraint != "latest":
                    self.logger.info(f"📋 Found version constraint in description for {package_name}: {description_constraint}")
                    return description_constraint
                else:
                    self.logger.info(f"📋 No version patterns found in description for {package_name}")
            
            # If we reach here, no version info found in vulnerability data
            self.logger.info(f"📋 No version constraint found in vulnerability data for {package_name} - will use spec fallback")
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
        """Extract version constraint using STRICT patterns - ONLY explicit upgrade instructions"""
        try:
            self.logger.info(f"🔍 DEBUG: Analyzing description for {package_name}")
            self.logger.info(f"🔍 DEBUG: Description text: '{description}'")
            self.logger.info(f"🔍 DEBUG: Description length: {len(description)}")
            
            # STRICT Pattern 1: ONLY Direct upgrade instructions (most explicit)
            strict_upgrade_patterns = [
                rf'[Uu]sers should upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Uu]sers should upgrade to ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Uu]sers should upgrade to {re.escape(package_name)} version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',  # NEW: Handle package name
                rf'[Uu]sers should upgrade to [A-Za-z\s]+ version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',  # NEW: Handle any package name
                rf'[Rr]ecommended to upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Mm]ust upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Pp]lease upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Ss]hould upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) to receive a fix',
                rf'upgrade to version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) to resolve',
                rf'upgrade to [A-Za-z\s]+ version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) or later',  # NEW: Handle "upgrade to Package version X.X.X or later"
            ]
            
            for i, pattern in enumerate(strict_upgrade_patterns):
                self.logger.info(f"🔍 DEBUG: Trying STRICT upgrade pattern {i+1}: {pattern}")
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    self.logger.info(f"📋 ✅ STRICT UPGRADE PATTERN MATCHED! Found explicit upgrade version for {package_name}: {version}")
                    return f">={version}"
            
            # STRICT Pattern 2: ONLY Clear fix/patch statements with explicit version
            strict_fixed_patterns = [
                rf'[Tt]his issue (?:has been|is) (?:patched|fixed|resolved) in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'(?:patched|fixed|resolved) in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Ff]ix (?:is )?available in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'[Pp]atch available in version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                rf'{re.escape(package_name)} ([0-9]+\.[0-9]+(?:\.[0-9]+)?) (?:fixes|resolves|patches) this',
            ]
            
            for i, pattern in enumerate(strict_fixed_patterns):
                self.logger.info(f"🔍 DEBUG: Trying STRICT fixed pattern {i+1}: {pattern}")
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    self.logger.info(f"📋 ✅ STRICT FIXED PATTERN MATCHED! Found explicit fix version for {package_name}: {version}")
                    return f">={version}"
            
            # REMOVED: All "prior to", "through", "before" patterns as they are descriptive, not prescriptive
            # These patterns were matching descriptive text like "NLTK through 3.8.1" which is NOT an upgrade instruction
            
            self.logger.info(f"📋 ❌ No STRICT explicit upgrade patterns matched for {package_name}")
            self.logger.info(f"📋 ❌ Description contains descriptive text, not explicit upgrade instructions")
            return "latest"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pattern matching failed for {package_name}: {e}")
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
            self.logger.info(f"🤖 Description length: {len(description)} chars")
            
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
        """Extract all Python vulnerabilities with ENHANCED regex patterns"""
        try:
            self.logger.info("🔍 Extracting all Safety vulnerabilities from new format")
            vulnerabilities = []
            seen_vulnerabilities = set()  # Track (package, vulnerability_id) pairs
            
            # ENHANCED: Multiple regex patterns to catch different formats
            
            # Pattern 1: Standard SafetyVulnerabilityAdvisory with double quotes
            pattern1 = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\] \[installed: ([^\]]+)\] \[vulnerabilities: \[(.*?)\]\]'
            matches1 = re.finditer(pattern1, logs, re.DOTALL)
            
            for match in matches1:
                package_name = match.group(1).strip()
                installed_version = match.group(2).strip()
                vulnerabilities_text = match.group(3).strip()
                
                self.logger.info(f"📋 Processing Safety report for package: {package_name} (installed: {installed_version})")
                self.logger.info(f"📋 Vulnerabilities section preview: {vulnerabilities_text[:200]}...")
                
                # ENHANCED: Multiple advisory patterns to catch different quote formats
                advisory_patterns = [
                    # Pattern A: Double quotes for advisory
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'[^)]*advisory="([^"]+)"[^)]*spec=\'([^\']+)\'[^)]*\)',
                    # Pattern B: Single quotes for advisory  
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'[^)]*advisory=\'([^\']+)\'[^)]*spec=\'([^\']+)\'[^)]*\)',
                    # Pattern C: No spec field (fallback)
                    r'SafetyVulnerabilityAdvisory\(vulnerability_id=\'([^\']+)\'[^)]*advisory="([^"]+)"[^)]*\)',
                    # Pattern D: More flexible order
                    r'vulnerability_id=\'([^\']+)\'[^,]*,[^,]*advisory="([^"]+)"[^,]*,[^,]*spec=\'([^\']+)\'',
                ]
                
                found_vulnerability = False
                for pattern_idx, advisory_pattern in enumerate(advisory_patterns):
                    advisory_matches = re.finditer(advisory_pattern, vulnerabilities_text, re.DOTALL)
                    
                    for advisory_match in advisory_matches:
                        try:
                            vuln_id = advisory_match.group(1).strip()
                            advisory_text = advisory_match.group(2).strip()
                            spec = advisory_match.group(3).strip() if len(advisory_match.groups()) >= 3 else 'unknown'
                            
                            # STRICT deduplication check
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
                    
                    # DEBUG: Try to find ANY vulnerability_id in the text
                    debug_vuln_ids = re.findall(r'vulnerability_id=\'([^\']+)\'', vulnerabilities_text)
                    if debug_vuln_ids:
                        self.logger.warning(f"❌ DEBUG: Found vulnerability IDs but couldn't parse: {debug_vuln_ids}")
                    else:
                        self.logger.warning(f"❌ DEBUG: No vulnerability_id patterns found at all")
            
            self.logger.info(f"📊 Total UNIQUE Safety vulnerabilities extracted: {len(vulnerabilities)}")
            
            # Enhanced debugging
            if len(vulnerabilities) == 0:
                self.logger.error("❌ ZERO vulnerabilities extracted! Debugging...")
                
                # Look for any SAFETY_REPORT lines
                safety_lines = re.findall(r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\]', logs)
                self.logger.error(f"❌ Found {len(safety_lines)} SAFETY_REPORT lines for packages: {safety_lines}")
                
                # Look for any vulnerability_id patterns anywhere
                all_vuln_ids = re.findall(r'vulnerability_id=\'([^\']+)\'', logs)
                self.logger.error(f"❌ Found {len(all_vuln_ids)} vulnerability_id patterns: {all_vuln_ids[:10]}")
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting Safety vulnerabilities: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def extract_complete_advisory_content(self, package_name: str, vulnerability_id: str, logs: str) -> str:
        """Extract advisory content from new SafetyVulnerabilityAdvisory format"""
        try:
            self.logger.info(f"🔍 Extracting advisory for {package_name} ({vulnerability_id}) from new Safety format")
            
            # Pattern 1: Extract from SafetyVulnerabilityAdvisory format with double quotes
            # Format: SafetyVulnerabilityAdvisory(vulnerability_id='77680', advisory="...", ...)
            safety_advisory_pattern = rf'SafetyVulnerabilityAdvisory\(vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory="([^"]+)"[^)]*\)'
            match = re.search(safety_advisory_pattern, logs, re.DOTALL)
            if match:
                advisory_content = match.group(1).strip()
                self.logger.info(f"📋 Pattern 1 SUCCESS: Found advisory via SafetyVulnerabilityAdvisory format ({len(advisory_content)} chars)")
                return advisory_content
            
            # Pattern 2: Try with single quotes (fallback)
            safety_advisory_pattern_single = rf'SafetyVulnerabilityAdvisory\(vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory=\'([^\']+)\'[^)]*\)'
            match = re.search(safety_advisory_pattern_single, logs, re.DOTALL)
            if match:
                advisory_content = match.group(1).strip()
                self.logger.info(f"📋 Pattern 2 SUCCESS: Found advisory via SafetyVulnerabilityAdvisory single quotes ({len(advisory_content)} chars)")
                return advisory_content
            
            # Pattern 3: Extract from the package-specific SAFETY_REPORT line
            # Find the entire SAFETY_REPORT line for this package and extract all advisories
            package_line_pattern = rf'SAFETY_REPORT \(FAILED\) \[pkg: {re.escape(package_name)}\].*?\[vulnerabilities: \[(.*?)\]\]'
            package_match = re.search(package_line_pattern, logs, re.DOTALL)
            if package_match:
                vulnerabilities_section = package_match.group(1)
                self.logger.info(f"📋 Pattern 3: Found vulnerabilities section for {package_name} ({len(vulnerabilities_section)} chars)")
                
                # Now find the specific vulnerability within this section
                vuln_in_section_pattern = rf'SafetyVulnerabilityAdvisory\(vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory="([^"]+)"[^)]*\)'
                vuln_match = re.search(vuln_in_section_pattern, vulnerabilities_section, re.DOTALL)
                if vuln_match:
                    advisory_content = vuln_match.group(1).strip()
                    self.logger.info(f"📋 Pattern 3 SUCCESS: Found advisory in package section ({len(advisory_content)} chars)")
                    return advisory_content
            
            # Pattern 4: Try to find any mention of this vulnerability ID anywhere and extract nearby advisory
            vuln_context_pattern = rf'vulnerability_id=\'{re.escape(vulnerability_id)}\'[^)]*advisory="([^"]+)"'
            match = re.search(vuln_context_pattern, logs, re.DOTALL)
            if match:
                advisory_content = match.group(1).strip()
                self.logger.info(f"📋 Pattern 4 SUCCESS: Found advisory via vulnerability ID search ({len(advisory_content)} chars)")
                return advisory_content
            
            # Pattern 5: Check original vulnerability data as fallback
            if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
                for pkg_name, pkg_vulns in self.original_vulnerability_data.items():
                    if isinstance(pkg_vulns, list):
                        for vuln in pkg_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                description = vuln.get('description', '')
                                if description:
                                    self.logger.info(f"📋 Pattern 5 SUCCESS: Found description in original data ({len(description)} chars)")
                                    return description
            
            self.logger.warning(f"❌ All patterns failed for {package_name} ({vulnerability_id})")
            return ""
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting advisory for {vulnerability_id}: {e}")
            return ""

    def get_optimal_version_constraint(self, package_name: str, vuln_data: Dict, all_logs: str, vulnerability_type: str = 'os_scan') -> str:
        """DEBUG VERSION: Show exactly what's happening with description extraction"""
        
        self.logger.info(f"🔍 Getting constraint for {package_name} ({vulnerability_type})")
        self.logger.info("="*80)
        self.logger.info(f"🎯 Checking advisory/description for explicit version info...")
        
        if vulnerability_type == 'py_scan':
            # For Python scan: extract FULL advisory content
            vulnerability_id = vuln_data.get('vulnerability_id', '')
            
            # Get the complete advisory
            full_advisory = self.extract_complete_advisory_content(package_name, vulnerability_id, all_logs)
            
            self.logger.info(f"🔍 DEBUG: Complete advisory for {package_name} ({vulnerability_id}):")
            self.logger.info(f"🔍 DEBUG: '{full_advisory}'")
            self.logger.info(f"🔍 DEBUG: Advisory length: {len(full_advisory)} characters")
            
            if full_advisory:
                # Look for explicit version in advisory
                version_constraint = self.extract_version_from_description_patterns(full_advisory, package_name)
                if version_constraint and version_constraint != "latest":
                    extracted_version = version_constraint.replace('>=', '').replace('>', '').replace('<', '').replace('=', '')
                    self.logger.info(f"📋 ✅ SUCCESS: Found explicit version in pyscan advisory!")
                    self.logger.info(f"📋 ✅ {package_name}: extracted version '{extracted_version}' → constraint {version_constraint}")
                    self.logger.info(f"📋 ✅ Will add to Dockerfile: {package_name}{version_constraint}")
                    self.logger.info("="*80)
                    return version_constraint
                else:
                    self.logger.info(f"📋 ❌ No explicit version found in pyscan advisory")
            else:
                self.logger.info(f"📋 ❌ Could not extract advisory content")
        
        else:
            # For OS scan: DEBUG ALL DESCRIPTION SOURCES
            self.logger.info(f"🔍 DEBUG: Checking OS scan description for {package_name}")
            
            # DEBUG: Show ALL available data sources
            self.logger.info("🔍 DEBUG: ========== VULN_DATA INSPECTION ==========")
            self.logger.info(f"🔍 DEBUG: vuln_data keys: {list(vuln_data.keys()) if isinstance(vuln_data, dict) else 'Not a dict'}")
            
            if 'original_data' in vuln_data:
                self.logger.info(f"🔍 DEBUG: original_data keys: {list(vuln_data['original_data'].keys()) if isinstance(vuln_data['original_data'], dict) else 'Not a dict'}")
                original_desc = vuln_data['original_data'].get('description', '')
                self.logger.info(f"🔍 DEBUG: original_data description length: {len(original_desc)}")
                self.logger.info(f"🔍 DEBUG: original_data description first 200 chars: '{original_desc[:200]}'")
                self.logger.info(f"🔍 DEBUG: original_data description last 200 chars: '{original_desc[-200:]}'")
            else:
                self.logger.info(f"🔍 DEBUG: No 'original_data' in vuln_data")
            
            direct_desc = vuln_data.get('description', '')
            self.logger.info(f"🔍 DEBUG: direct description length: {len(direct_desc)}")
            self.logger.info(f"🔍 DEBUG: direct description first 200 chars: '{direct_desc[:200]}'")
            self.logger.info(f"🔍 DEBUG: direct description last 200 chars: '{direct_desc[-200:]}'")
            
            # Get the full description from original vulnerability data
            full_description = ""
            if 'original_data' in vuln_data and vuln_data['original_data']:
                full_description = vuln_data['original_data'].get('description', '')
                self.logger.info(f"🔍 DEBUG: Using original_data description ({len(full_description)} chars)")
            if not full_description:
                full_description = vuln_data.get('description', '')
                self.logger.info(f"🔍 DEBUG: Fallback to direct description ({len(full_description)} chars)")
                
            self.logger.info(f"🔍 DEBUG: ========== FINAL DESCRIPTION TO ANALYZE ==========")
            self.logger.info(f"🔍 DEBUG: Complete OS description for {package_name}:")
            self.logger.info(f"🔍 DEBUG: Length: {len(full_description)} characters")
            self.logger.info(f"🔍 DEBUG: Full text: '{full_description}'")
            
            # Check if the description contains expected keywords
            upgrade_keywords = ['upgrade', 'version', '5.8.0', 'patch']
            found_keywords = [kw for kw in upgrade_keywords if kw in full_description.lower()]
            self.logger.info(f"🔍 DEBUG: Found keywords: {found_keywords}")
            
            # Check for the specific jupyter_core pattern
            if package_name.lower() == 'jupyter_core' or 'jupyter' in package_name.lower():
                self.logger.info(f"🔍 DEBUG: ========== JUPYTER_CORE SPECIFIC DEBUG ==========")
                test_patterns = [
                    r'[Uu]sers should upgrade to.*?version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                    r'upgrade to.*?version ([0-9]+\.[0-9]+(?:\.[0-9]+)?)',
                    r'version ([0-9]+\.[0-9]+(?:\.[0-9]+)?) or later',
                    r'([0-9]+\.[0-9]+(?:\.[0-9]+)?)',  # Any version number
                ]
                
                for i, pattern in enumerate(test_patterns):
                    matches = re.findall(pattern, full_description, re.IGNORECASE)
                    self.logger.info(f"🔍 DEBUG: Test pattern {i+1} '{pattern}' found: {matches}")
            
            if full_description:
                # DIRECT pattern matching
                version_constraint = self.extract_version_from_description_patterns(full_description, package_name)
                if version_constraint and version_constraint != "latest":
                    extracted_version = version_constraint.replace('>=', '').replace('>', '').replace('<', '').replace('=', '')
                    self.logger.info(f"📋 ✅ SUCCESS: Found explicit version in OS scan description!")
                    self.logger.info(f"📋 ✅ {package_name}: extracted version '{extracted_version}' → constraint {version_constraint}")
                    self.logger.info(f"📋 ✅ Will add to Dockerfile: {package_name}{version_constraint}")
                    self.logger.info("="*80)
                    return version_constraint
                else:
                    self.logger.info(f"📋 ❌ No explicit version found in OS scan description")
            else:
                self.logger.info(f"📋 ❌ No OS scan description available")
        
        # No explicit version found - go directly to allowlist
        self.logger.info(f"📋 ⚠️ DECISION: No explicit version found for {package_name}")
        self.logger.info(f"📋 ⚠️ Will skip Dockerfile and add {package_name} to allowlist")
        self.logger.info("="*80)
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
    
    def preprocess_logs_for_ai(self, logs: str) -> str:
        """Preprocess logs and preserve original vulnerability data for exact allowlist formatting"""
        self.logger.info("🔧 Preprocessing logs for better AI analysis...")
        
        processed_logs = logs
        
        # Store original vulnerability data for later use (existing logic for OS scan)
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

        # NEW: Extract Python scan data using the new Safety format
        self.logger.info("🔍 Extracting Python vulnerabilities from new Safety format...")
        py_vulnerabilities = self.extract_all_safety_vulnerabilities_from_logs(logs)
        
        if py_vulnerabilities:
            processed_section = f"""

    === PYTHON SCAN VULNERABILITIES (New Safety Format) ===
    {json.dumps(py_vulnerabilities, indent=2)}
    === END PYTHON SCAN VULNERABILITIES ===
    """
            processed_logs += processed_section
            self.logger.info(f"✅ Extracted {len(py_vulnerabilities)} Python vulnerabilities from new format")
        
        self.logger.info(f"🔧 Preprocessing complete. Stored original data for {len(self.original_vulnerability_data)} packages")
        return processed_logs
    
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
        
        # Handle Python vulnerabilities with FIXED REGEX
        # FIXED: Same regex fix as in preprocess_logs_for_ai
        safety_pattern = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'(?:.*?advisory=\'([^\']*?)\')?'
        safety_matches = list(re.finditer(safety_pattern, logs, re.DOTALL))
        
        seen_vulnerabilities = set()
        for match in safety_matches:
            groups = match.groups()
            package = groups[0].strip()
            vuln_id = groups[1].strip()
            advisory = groups[2].strip() if groups[2] else f'Security vulnerability in {package}'
            
            # Log the extracted advisory for debugging
            self.logger.info(f"🔍 FALLBACK: Extracted advisory for {package} ({vuln_id}): '{advisory[:100]}{'...' if len(advisory) > 100 else ''}'")
            
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

    def ai_detect_vulnerabilities_only_with_retry(self, logs: str, max_retries: int = 3) -> Dict:
        """Use AI only for vulnerability detection - MATCH WITH COMPLETE ORIGINAL DATA"""
        self.logger.info("🤖 Using AI only for vulnerability detection, matching with complete original data...")
        
        try:
            # Truncate logs for AI processing
            truncated_logs = self.truncate_logs_for_ai(logs, max_chars=50000)
            
            # Preprocess logs to help AI detection AND store original data
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
            
            # Get detected vulnerabilities
            ai_os_vulns = detected.get('os_vulns', [])
            ai_py_vulns = detected.get('py_vulns', [])
            
            self.logger.info(f"🤖 AI detected {len(ai_os_vulns)} OS and {len(ai_py_vulns)} Python vulnerabilities")
            
            # ENHANCED: Match AI detections with complete original data
            enhanced_os_vulns = []
            enhanced_py_vulns = []
            
            # Process OS vulnerabilities - match with original data
            for ai_vuln in ai_os_vulns:
                package = ai_vuln.get('package', '').lower()
                vuln_id = ai_vuln.get('vulnerability_id', '')
                
                # Find complete original vulnerability data
                original_data = self.find_original_vulnerability_data(package, vuln_id)
                
                if original_data:
                    # Create enhanced vulnerability with COMPLETE original data
                    enhanced_vuln = original_data.copy()  # Start with ALL original fields
                    
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
                    
                    # Verify key fields are preserved
                    key_fields = ['cvss_v3_score', 'cvss_v31_score', 'cvss_v3_severity', 'severity']
                    preserved_fields = [f for f in key_fields if f in enhanced_vuln and enhanced_vuln[f] not in [None, 0.0, 'UNKNOWN', '']]
                    missing_fields = [f for f in key_fields if f not in enhanced_vuln or enhanced_vuln[f] in [None, 0.0, 'UNKNOWN', '']]
                    
                    self.logger.info(f"✅ Enhanced {vuln_id} in {package}")
                    self.logger.info(f"  ✅ Preserved fields: {preserved_fields}")
                    if missing_fields:
                        self.logger.warning(f"  ⚠️ Missing/empty fields: {missing_fields}")
                        
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
            
            # Process Python vulnerabilities (typically don't have original structured data)
            for ai_vuln in ai_py_vulns:
                enhanced_py_vulns.append({
                    'package': ai_vuln.get('package', ''),
                    'vulnerability_id': ai_vuln.get('vulnerability_id', ''),
                    'description': ai_vuln.get('description', ''),
                    'severity': ai_vuln.get('severity', 'UNKNOWN')
                })
            
            self.logger.info(f"🔍 ENHANCED RESULT: {len(enhanced_os_vulns)} OS (with complete data), {len(enhanced_py_vulns)} Python vulnerabilities")
            
            return {
                'os_vulns': enhanced_os_vulns,      # Now with complete original data
                'py_vulns': enhanced_py_vulns       
            }
            
        except Exception as e:
            self.logger.error(f"❌ AI detection failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'os_vulns': [], 'py_vulns': []}
    ## 2. New `ai_detect_vulnerabilities_only()` method:

    def ai_detect_vulnerabilities_only(self, logs: str) -> Dict:
        """Main entry point - delegates to retry version"""
        return self.ai_detect_vulnerabilities_only_with_retry(logs, max_retries=3)

        
    def extract_all_vulnerabilities(self, logs: str) -> Dict:
        """Extract all vulnerabilities using AI for detection only, then rule-based formatting - PRESERVE COMPLETE DATA"""
        self.logger.info("🤖 Using AI only for vulnerability detection, preserving complete original data...")
        
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
            
            # Step 3: Match detected OS vulnerabilities with COMPLETE original data 
            for detected_vuln in detected_vulnerabilities.get('os_vulns', []):
                package = detected_vuln.get('package', '').lower()
                vuln_id = detected_vuln.get('vulnerability_id', '')
                
                # Find COMPLETE original vulnerability data
                original_data = self.find_original_vulnerability_data(package, vuln_id)
                
                if original_data:
                    # Use the COMPLETE original data structure, but update description if AI found a better one
                    complete_vuln = original_data.copy()  # Start with all original fields
                    
                    # Only update description if AI found upgrade information that wasn't in original
                    ai_description = detected_vuln.get('description', '')
                    original_description = original_data.get('description', '')
                    
                    # Use AI description if it contains upgrade info that original doesn't have
                    if ('upgrade' in ai_description.lower() and 'version' in ai_description.lower() and
                        not ('upgrade' in original_description.lower() and 'version' in original_description.lower())):
                        complete_vuln['description'] = ai_description
                        self.logger.info(f"📋 Using AI description with upgrade info for {vuln_id}")
                    
                    # Ensure we have the key fields for processing
                    complete_vuln.update({
                        'package': package,
                        'vulnerability_id': vuln_id,
                        'original_data': original_data  # Store complete original for allowlist formatting
                    })
                    
                    os_vulnerabilities.append(complete_vuln)
                    self.logger.info(f"✅ Matched OS vulnerability with COMPLETE original data: {vuln_id} in {package}")
                    
                    # Verify key fields are preserved
                    key_fields = ['cvss_v3_score', 'cvss_v31_score', 'cvss_v3_severity', 'severity']
                    for field in key_fields:
                        if field in complete_vuln:
                            self.logger.info(f"  ✅ Preserved {field}: {complete_vuln[field]}")
                        else:
                            self.logger.warning(f"  ❌ Missing {field} in complete vulnerability")
                            
                else:
                    # Fallback if no original data found - use AI detection but with minimal fields
                    fallback_vuln = {
                        'package': package,
                        'vulnerability_id': vuln_id,
                        'description': detected_vuln.get('description', f'Vulnerability {vuln_id}'),
                        'severity': detected_vuln.get('severity', 'UNKNOWN'),
                        'original_data': {}
                    }
                    os_vulnerabilities.append(fallback_vuln)
                    self.logger.warning(f"⚠️ Using fallback data for OS vulnerability: {vuln_id} in {package}")
            
            # Handle Python vulnerabilities (simpler structure, usually no original data)
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
        """Find original vulnerability data from preprocessed logs - PRESERVE ALL FIELDS"""
        try:
            if hasattr(self, 'original_vulnerability_data') and self.original_vulnerability_data:
                package_key = package_name.lower()
                
                # Look for the package in original data
                if package_key in self.original_vulnerability_data:
                    package_vulns = self.original_vulnerability_data[package_key]
                    
                    if isinstance(package_vulns, list):
                        for vuln in package_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                self.logger.info(f"📋 Found complete original data for {vulnerability_id} in {package_name}")
                                
                                # Verify key fields are present
                                key_fields = ['cvss_v3_score', 'cvss_v31_score', 'cvss_v3_severity', 'severity', 'package_details']
                                present_fields = [f for f in key_fields if f in vuln and vuln[f] not in [None, 0.0, 'UNKNOWN', '']]
                                self.logger.info(f"✅ Original data has fields: {present_fields}")
                                
                                return vuln  # Return the COMPLETE original structure
                    
                # If exact package name not found, try variations
                for stored_package, stored_vulns in self.original_vulnerability_data.items():
                    if isinstance(stored_vulns, list):
                        for vuln in stored_vulns:
                            if isinstance(vuln, dict) and vuln.get('vulnerability_id') == vulnerability_id:
                                # Check if package names are similar
                                if (package_name.lower() in stored_package.lower() or 
                                    stored_package.lower() in package_name.lower()):
                                    self.logger.info(f"📋 Found complete original data for {vulnerability_id} via package match: {stored_package}")
                                    return vuln  # Return the COMPLETE original structure
                
                self.logger.warning(f"⚠️ No original vulnerability data found for {vulnerability_id} in {package_name}")
                return {}
                
            else:
                self.logger.warning(f"⚠️ No original_vulnerability_data available")
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Error finding original vulnerability data: {e}")
            return {}
        
    def extract_all_vulnerabilities_by_container(self, all_logs: str) -> Dict:
        """Extract vulnerabilities organized by container type WITHOUT global deduplication"""
        self.logger.info("🤖 Extracting vulnerabilities by container type (allowing per-container duplicates)...")
        
        # Initialize result structure
        result = {
            'training': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'inference': {'os_vulnerabilities': [], 'py_vulnerabilities': []},
            'unknown': {'os_vulnerabilities': [], 'py_vulnerabilities': []}
        }
        
        # Process each container type's logs separately
        container_order = ['training', 'inference', 'unknown']  # Define order for processing
        
        for i, container_type in enumerate(container_order):
            logs = self.container_specific_logs.get(container_type, '')
            
            if not logs.strip():
                self.logger.info(f"📊 No logs for {container_type} container")
                continue
                
            self.logger.info(f"📊 Processing {container_type} container logs ({len(logs)} chars)")
            
            # Add 5-minute delay before inference processing to avoid throttling
            if container_type == 'inference' and i > 0:  # Only delay if not the first container
                self.logger.info("⏳ Adding 5-minute delay before inference AI processing to avoid throttling...")
                delay_seconds = 5 * 60  # 5 minutes
                
                # Show countdown
                for remaining in range(delay_seconds, 0, -30):  # Show every 30 seconds
                    minutes = remaining // 60
                    seconds = remaining % 60
                    self.logger.info(f"⏳ Waiting {minutes}:{seconds:02d} before inference processing...")
                    time.sleep(30)
                
                self.logger.info("✅ Delay complete, proceeding with inference AI processing...")
            
            # Container-specific deduplication (only within this container)
            container_seen_os = set()  # Track (package, vuln_id) for this container only
            container_seen_py = set()  # Track (package, vuln_id) for this container only
            
            # Extract vulnerabilities for this container type with retry
            try:
                container_vulns = self.ai_detect_vulnerabilities_only_with_retry(logs)
                
                # Add container type to each vulnerability with CONTAINER-SPECIFIC deduplication
                for vuln in container_vulns.get('os_vulns', []):
                    vuln['container_type'] = container_type
                    
                    # Container-specific deduplication for OS vulns
                    unique_key = (vuln['package'], vuln['vulnerability_id'])
                    if unique_key not in container_seen_os:
                        container_seen_os.add(unique_key)
                        result[container_type]['os_vulnerabilities'].append(vuln)
                        self.logger.info(f"✅ NEW OS ({container_type}): {vuln['package']} - {vuln['vulnerability_id']}")
                    else:
                        self.logger.info(f"⚠️ DUPLICATE OS WITHIN {container_type}: {vuln['package']} - {vuln['vulnerability_id']}")
                        
                for vuln in container_vulns.get('py_vulns', []):
                    vuln['container_type'] = container_type
                    
                    # Container-specific deduplication for Python vulns
                    unique_key = (vuln['package'], vuln['vulnerability_id'])
                    if unique_key not in container_seen_py:
                        container_seen_py.add(unique_key)
                        result[container_type]['py_vulnerabilities'].append(vuln)
                        self.logger.info(f"✅ NEW PY ({container_type}): {vuln['package']} - {vuln['vulnerability_id']}")
                    else:
                        self.logger.info(f"⚠️ DUPLICATE PY WITHIN {container_type}: {vuln['package']} - {vuln['vulnerability_id']}")
                        
                self.logger.info(f"📊 {container_type}: {len(result[container_type]['os_vulnerabilities'])} OS, {len(result[container_type]['py_vulnerabilities'])} Python vulnerabilities (container-specific dedup)")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to extract vulnerabilities for {container_type}: {e}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Store in instance variable for later use
        self.container_specific_vulnerabilities = result
        
        # Log final summary - now showing vulnerabilities can appear in multiple containers
        total_os = sum(len(container['os_vulnerabilities']) for container in result.values())
        total_py = sum(len(container['py_vulnerabilities']) for container in result.values())
        self.logger.info(f"📊 FINAL: {total_os} total OS, {total_py} total Python vulnerabilities across all containers")
        self.logger.info("ℹ️ Note: Same vulnerability can appear in multiple containers and will be handled separately")
        
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
        
        # Extract Python vulnerabilities with FIXED REGEX
        # FIXED: Same regex fix as in other methods
        safety_pattern = r'SAFETY_REPORT \(FAILED\) \[pkg: ([^\]]+)\].*?vulnerability_id=\'([^\']+)\'(?:.*?advisory=\'([^\']*?)\')?'        
        safety_matches = list(re.finditer(safety_pattern, logs, re.DOTALL))
        
        seen_py_vulns = set()
        for match in safety_matches:
            groups = match.groups()
            package = groups[0].strip()
            vuln_id = groups[1].strip()
            advisory = groups[2].strip() if groups[2] else f'Security vulnerability in {package}'
            
            # Log the extracted advisory for debugging
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
        """Simplified version with upgrade check - just calls the main constraint method"""
        constraint = self.get_optimal_version_constraint(package_name, vuln_data, all_logs, vulnerability_type)
        
        # No need for upgrade checks anymore since we either have explicit version or go to allowlist
        return constraint

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
        """Apply Dockerfile fixes only to relevant container types - FIXED DATA PASSING"""
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
                # FIXED: Pass the complete vulnerability data, not just original_data
                vuln_data_to_pass = vuln.get('original_data', {})
                if not vuln_data_to_pass:
                    # If no original_data, create it from the detected vulnerability
                    vuln_data_to_pass = {
                        'description': vuln.get('description', ''),
                        'vulnerability_id': vuln_id,
                        'package_name': package,
                        'severity': vuln.get('severity', 'UNKNOWN')
                    }
                    self.logger.info(f"🔍 Created vuln_data for {package} from AI detection: {len(vuln_data_to_pass.get('description', ''))} chars")
                
                # Get optimal version constraint for OS scan (description parsing only)
                version_constraint = self.get_optimal_version_constraint_with_upgrade_check(
                    package, 
                    vuln_data_to_pass,  # <-- FIXED: Pass complete data
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
                    vuln,  # Pass the complete vulnerability for Python scan
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


    def extract_current_version_from_logs(self, package_name: str, logs: str) -> str:
        """Extract current installed version from Safety report logs"""
        try:
            
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
        """Increment version number safely - ENHANCED"""
        try:
            parts = version.split('.')
            if len(parts) >= 3:
                major = int(parts[0])
                minor = int(parts[1]) 
                patch = int(parts[2])
                
                if increment_patch:
                    return f"{major}.{minor}.{patch + 1}"
                else:
                    return f"{major}.{minor + 1}.0"
            elif len(parts) == 2:
                major = int(parts[0])
                minor = int(parts[1])
                
                if increment_patch:
                    return f"{major}.{minor}.1"
                else:
                    return f"{major}.{minor + 1}.0"
            else:
                # Single number version
                return f"{int(version) + 1}.0.0"
        except (ValueError, IndexError):
            self.logger.warning(f"⚠️ Could not increment version {version}")
            return version
    def parse_safety_spec_to_constraint(self, spec: str, package_name: str, logs: str) -> str:
        """Parse Safety spec into proper pip constraint - FIXED LOGIC"""
        try:
            self.logger.info(f"📋 Parsing Safety spec '{spec}' for {package_name}")
            
            if spec.startswith('<'):
                # <X.X.X means vulnerable in versions < X.X.X
                # So fix is >= X.X.X
                fix_version = spec[1:].strip()
                constraint = f">={fix_version}"
                self.logger.info(f"📋 Safety spec '{spec}' → FIX: {constraint}")
                return constraint
            
            elif spec.startswith('<='):
                # <=X.X.X means vulnerable in versions <= X.X.X  
                # So fix is > X.X.X (increment patch version)
                max_version = spec[2:].strip()
                next_version = self.increment_version(max_version, increment_patch=True)
                constraint = f">={next_version}"
                self.logger.info(f"📋 Safety spec '{spec}' → FIX: {constraint}")
                return constraint
            
            elif spec.startswith('>='):
                # >=X.X.X means already the minimum version
                min_version = spec[2:].strip()
                return f">={min_version}"
            
            elif spec.startswith('>'):
                # >X.X.X means need higher than X.X.X
                min_version = spec[1:].strip()
                return f">{min_version}"
            
            elif ',' in spec:
                # Complex constraint, use as-is
                return spec
            
            else:
                # Exact version
                return f"=={spec}"
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not parse Safety spec '{spec}': {e}")
            return "latest"

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