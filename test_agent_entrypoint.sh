#!/bin/bash
set -e

echo "🧪 AutoGluon Test Agent - Starting Execution"
echo "================================================"

# Print environment info
echo "📋 Environment Information:"
echo "   Current Version: $currentVersion"
echo "   Previous Version: $previousVersion"
echo "   Fork URL: $forkUrl"
echo "   Job Parameters: $jobParams"
echo "   Account ID: $ACCOUNT_ID"
echo "   Region: $REGION"
echo "   Working Directory: $(pwd)"
echo "   User: $(whoami)"
echo "   UID: $(id -u)"
echo "   GID: $(id -g)"
echo ""

# Enhanced debugging - check file system state
echo "🔍 File System Debug Information:"
echo "   Files in /app:"
ls -la /app/
echo ""
echo "   Files in /app/autogluon_test_files:"
if [ -d "/app/autogluon_test_files" ]; then
    ls -la /app/autogluon_test_files/
    echo ""
    echo "   Test file permissions and contents:"
    for file in /app/autogluon_test_files/*.py; do
        if [ -f "$file" ]; then
            echo "     $file:"
            echo "       - Permissions: $(ls -la "$file")"
            echo "       - Size: $(wc -l < "$file") lines"
            echo "       - First line: $(head -n 1 "$file")"
        fi
    done
else
    echo "   ❌ /app/autogluon_test_files directory not found!"
fi
echo ""

# Validate required environment variables
if [ -z "$currentVersion" ]; then
    echo "❌ ERROR: currentVersion environment variable not set"
    exit 1
fi

if [ -z "$previousVersion" ]; then
    echo "❌ ERROR: previousVersion environment variable not set"
    exit 1
fi

if [ -z "$forkUrl" ]; then
    echo "❌ ERROR: forkUrl environment variable not set"
    exit 1
fi

if [ -z "$ACCOUNT_ID" ]; then
    echo "❌ ERROR: ACCOUNT_ID environment variable not set"
    exit 1
fi

# Set default values
REGION=${REGION:-us-east-1}

# Parse job parameters to extract test-specific options
TEST_SUITE="all"
IMAGE_TAG=""

if [ ! -z "$jobParams" ]; then
    echo "🔍 Parsing job parameters: $jobParams"
    
    # Extract test suite if specified
    if [[ $jobParams == *"--test-suite="* ]]; then
        TEST_SUITE=$(echo "$jobParams" | grep -o '\--test-suite=[^[:space:]]*' | cut -d'=' -f2)
        echo "   Test Suite: $TEST_SUITE"
    fi
    
    # Extract image tag if specified
    if [[ $jobParams == *"--image-tag="* ]]; then
        IMAGE_TAG=$(echo "$jobParams" | grep -o '\--image-tag=[^[:space:]]*' | cut -d'=' -f2)
        echo "   Image Tag: $IMAGE_TAG"
    fi
fi

# Prepare test agent command
CMD_ARGS="--current-version $currentVersion --previous-version $previousVersion --fork-url $forkUrl"

# Add test suite if not default
if [ "$TEST_SUITE" != "all" ]; then
    CMD_ARGS="$CMD_ARGS --test-suite $TEST_SUITE"
fi

# Add image tag if specified
if [ ! -z "$IMAGE_TAG" ]; then
    CMD_ARGS="$CMD_ARGS --image-tag $IMAGE_TAG"
fi

echo ""
echo "🚀 Executing AutoGluon Test Agent"
echo "   Command: python autogluon_test_agent.py $CMD_ARGS"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /app/logs/autogluon_tests

# Set up logging
export PYTHONUNBUFFERED=1

# Verify test files exist with enhanced checking
echo "📂 Detailed test files verification..."
if [ ! -d "/app/autogluon_test_files" ]; then
    echo "❌ ERROR: Test files directory not found at /app/autogluon_test_files"
    echo "   Searching for test files in other locations..."
    find /app -name "test_*.py" -type f 2>/dev/null || echo "   No test files found anywhere in /app"
    exit 1
fi

echo "✅ Test files directory found: /app/autogluon_test_files"

# Check each required test file
required_files=("test_automm.py" "test_tabular.py" "test_ts.py")
for file in "${required_files[@]}"; do
    filepath="/app/autogluon_test_files/$file"
    if [ ! -f "$filepath" ]; then
        echo "❌ ERROR: Required test file not found: $filepath"
        exit 1
    fi
    
    # Check if file is readable
    if [ ! -r "$filepath" ]; then
        echo "❌ ERROR: Test file is not readable: $filepath"
        echo "   Permissions: $(ls -la "$filepath")"
        exit 1
    fi
    
    # Check if file has content
    if [ ! -s "$filepath" ]; then
        echo "❌ ERROR: Test file is empty: $filepath"
        exit 1
    fi
    
    echo "✅ Test file OK: $file ($(wc -l < "$filepath") lines)"
done

echo ""
echo "🔧 Verifying AWS credentials..."
aws sts get-caller-identity || {
    echo "❌ ERROR: AWS credentials not properly configured"
    exit 1
}

echo "✅ AWS credentials verified"

echo ""
echo "🐳 Verifying Docker access..."
docker --version || {
    echo "❌ ERROR: Docker not available"
    exit 1
}

echo "✅ Docker verified"

# Test Docker socket access
echo "🔍 Testing Docker socket access..."
docker ps > /dev/null 2>&1 || {
    echo "❌ ERROR: Cannot access Docker daemon"
    echo "   This might be expected in some environments"
    echo "   Continuing with test agent execution..."
}

echo ""
echo "🔐 Authenticating Docker with ECR..."
# Login to ECR so Docker can pull training images
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

if [ $? -eq 0 ]; then
    echo "✅ ECR authentication successful"
else
    echo "⚠️ ECR authentication failed, but continuing..."
fi

echo ""
echo "🐍 Python environment check..."
echo "   Python version: $(python --version)"
echo "   Python path: $(which python)"
echo "   Pip packages:"
pip list | grep -E "(boto3|pathlib|requests)" || echo "   Some packages may not be installed"

echo ""
echo "================================================"
echo "🧪 Starting AutoGluon Test Agent Execution"
echo "================================================"

# Execute the test agent with enhanced error handling
set +e  # Don't exit on error so we can capture and report it
python autogluon_test_agent.py $CMD_ARGS

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ AutoGluon Test Agent completed successfully!"
else
    echo "❌ AutoGluon Test Agent failed with exit code: $EXIT_CODE"
    echo ""
    echo "🔍 Post-failure debugging:"
    echo "   Final working directory: $(pwd)"
    echo "   Final test files state:"
    ls -la /app/autogluon_test_files/ || echo "   Test files directory not accessible"
    echo "   Recent log files:"
    find /app/logs -name "*.log" -type f -exec echo "   {}: $(tail -n 3 {})" \; 2>/dev/null || echo "   No log files found"
fi
echo "================================================"

# Exit with the same code as the test agent
exit $EXIT_CODE