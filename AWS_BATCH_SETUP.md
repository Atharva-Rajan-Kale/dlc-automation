# AWS Batch Integration Setup

## Required GitHub Secrets

Add these secrets to your GitHub repository settings:

### AWS Credentials
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key  
- `AWS_SESSION_TOKEN` - AWS session token (if using temporary credentials)
- `ACCOUNT_ID` - AWS account ID

### CodeBuild Credentials (if different from main AWS credentials)
- `CODEBUILD_ACCESS_KEY_ID`
- `CODEBUILD_SECRET_ACCESS_KEY`
- `CODEBUILD_AWS_SESSION_TOKEN`

### Bedrock Credentials
- `BEDROCK_INFERENCE_PROFILE_ARN` - ARN for Bedrock inference profile

### AWS Batch Configuration
- `BATCH_JOB_QUEUE` - Name of your AWS Batch job queue
- `BATCH_JOB_DEFINITION` - Job definition for main automation tasks
- `BATCH_TEST_JOB_DEFINITION` - Job definition for testing tasks
- `BATCH_INFRA_JOB_DEFINITION` - Job definition for infrastructure tasks
- `BATCH_RELEASE_JOB_DEFINITION` - Job definition for release tasks
- `BATCH_SECURITY_JOB_DEFINITION` - Job definition for security tasks
- `BATCH_RESULTS_BUCKET` - S3 bucket for storing job results

### GitHub Token
- `GITHUB_TOKEN` - GitHub personal access token

## AWS Batch Job Definitions

Your job definitions should accept these parameters:
- `currentVersion` - Current version being processed
- `previousVersion` - Previous version for comparison
- `forkUrl` - GitHub fork URL
- `testType` - Type of test to run (for test jobs)
- `analysisType` - Type of analysis (for security jobs)
- `jobParams` - Additional job-specific parameters
- `skipConfirmations` - Whether to skip manual confirmations
- `prNumber` - PR number (for security analysis)

## Container Requirements

Your Docker containers should:
1. Have all required dependencies pre-installed
2. Accept parameters via environment variables
3. Upload results to the specified S3 bucket
4. Exit with appropriate status codes (0 for success, non-zero for failure)