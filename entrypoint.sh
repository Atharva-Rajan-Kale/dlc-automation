#!/bin/bash
set -e

# Get parameters from AWS Batch with defaults
CURRENT_VERSION=${currentVersion:-"1.4.0"}
PREVIOUS_VERSION=${previousVersion:-"1.3.0"}
FORK_URL=${forkUrl:-"https://github.com/test/repo"}
JOB_PARAMS=${jobParams:-"--complete"}

echo "Starting automation with version: $PREVIOUS_VERSION → $CURRENT_VERSION"

# Run the main automation script
if [ -n "$JOB_PARAMS" ] && [ "$JOB_PARAMS" != "\$JOB_PARAMS" ]; then
  python enhanced_main_automation.py \
    --current-version="$CURRENT_VERSION" \
    --previous-version="$PREVIOUS_VERSION" \
    --fork-url="$FORK_URL" \
    $JOB_PARAMS
else
  python enhanced_main_automation.py \
    --current-version="$CURRENT_VERSION" \
    --previous-version="$PREVIOUS_VERSION" \
    --fork-url="$FORK_URL" \
    --complete
fi

# Upload results to S3
if [ -d "automation_logs" ]; then
  aws s3 cp automation_logs/ s3://${BATCH_RESULTS_BUCKET}/automation-${AWS_BATCH_JOB_ID}/ --recursive
fi

if [ -d "step_outputs" ]; then
  aws s3 cp step_outputs/ s3://${BATCH_RESULTS_BUCKET}/automation-${AWS_BATCH_JOB_ID}/ --recursive
fi