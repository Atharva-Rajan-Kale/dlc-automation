#!/bin/bash
set -e

# Get parameters from AWS Batch
TEST_TYPE=${testType}
CURRENT_VERSION=${currentVersion}
PREVIOUS_VERSION=${previousVersion}
FORK_URL=${forkUrl}

echo "Running test: $TEST_TYPE"

# Run appropriate test based on type
case "$TEST_TYPE" in
  "pip-check")
    python pip_check_agent.py --current-version="$CURRENT_VERSION" --previous-version="$PREVIOUS_VERSION" --fork-url="$FORK_URL"
    ;;
  "autogluon-tests")
    python autogluon_test_agent.py --current-version="$CURRENT_VERSION" --previous-version="$PREVIOUS_VERSION" --fork-url="$FORK_URL"
    ;;
  "sagemaker-tests")
    python sagemaker_test_agent.py --current-version="$CURRENT_VERSION" --previous-version="$PREVIOUS_VERSION" --fork-url="$FORK_URL"
    ;;
  "security-tests")
    python security_test_agent.py --current-version="$CURRENT_VERSION" --previous-version="$PREVIOUS_VERSION" --fork-url="$FORK_URL"
    ;;
  "quick-checks")
    python quick_checks_agent.py --current-version="$CURRENT_VERSION" --previous-version="$PREVIOUS_VERSION" --fork-url="$FORK_URL"
    ;;
  *)
    echo "Unknown test type: $TEST_TYPE"
    exit 1
    ;;
esac

# Upload test results to S3
if [ -d "automation_logs" ]; then
  aws s3 cp automation_logs/ s3://${BATCH_RESULTS_BUCKET}/test-${TEST_TYPE}-${AWS_BATCH_JOB_ID}/ --recursive
fi