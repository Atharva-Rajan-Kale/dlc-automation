# Deep Learning Container Automation

An intelligent automation system for Autogluon DLC release that involves file updation, building docker images, dependency management, security testing, and CI/CD pipeline integration using AI-powered analysis.

## Architecture Overview
This automation system consists of several integrated agents working together:

### Core Components: 

1. Security Test Agent: Monitors and analyzes security test failures, collecting logs and parsing by container type.
2. Pip Check Agent: Manages Python package dependencies and version conflicts.
3. AI Analysis Engine: Leverages Claude AI (via Langchain + Bedrock) for intelligent vulnerability extraction and conflict resolution. 
4. CI/CD Integration: Seamless integration with Asimov CI PR Testing and automated workflows.

## Prerequisites

### Running automation locally

1. AWS Credentials: Valid AWS credentials with appropriate permissions.
2. Fork Repository: Fork the main deep-learning-containers repository. Name it as deep-learning-container on your github.
3. Asimov AWS Credentials: Credentials for Asimov CI/PR testing. Name it in the following way: \
CODEBUILD_AWS_SECRET_KEY,etc. \
Also define CODEBUILD_REGION='us-west-2'
4. GitHub Token: Personal access token for repository operations.
5. Bedrock ARN: For claude access
6. ACCOUNT_ID, REGION of your ECR repository.

### Running automation via CI/CD

Configure the above credentials in your github secrets.

Setup AWS Batch with the following:

1. Job definition referring to the docker image of the codebase.
2. Job queue referring to the compute environment being used.
3. Compute environment with the appropriate instance type and memory usage for the job.
4. IAM roles for docker building, access to ECR, and running the images.

## Workflow Process

1. Input Processing: Handles version comparisons and forked repository URLs.
2. Container Analysis: Creates and updates Docker files, builds images, and executes git operations.
3. Dependency Management: Performs pip checks, version matching, and conflict resolution.
4. Automated Testing: Executes autologen tests and sagemaker tests.
5. Security Testing: Runs comprehensive security scans and vulnerability assessments.
6. Error Resolution: AI-powered error detection and extraction from logs.


## Installation

### Clone the Repository
```shell script
git clone https://github.com/Atharva-Rajan-Kale/dlc-automation.git
cd dlc-automation
```

### Setup
```shell script
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip
```

### Install Requirements
```shell script
pip install -r requirements_agent.txt
```
### Complete Workflow Options

#### 1. Full Automation (Default)
   Runs the complete end-to-end workflow :
   File updation -> Docker Build -> Pip Check and rebuild if necessary -> Autogluon Test -> Sagemaker tests -> PR Creation
   
```shell script
python -m automation.enhanced_main_automation \
    --current-version <current_version> \
    --previous-version <previous_version> \
    --fork-url <your_forked_repository_url>
```

**Example Usage** 
```shell script
python -m automation.enhanced_main_automation \
    --current-version 1.3.1 \
    --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git
```

### Individual Step Groups

#### 1. Runs the initial steps before docker image building
This includes cloning the repository, updating toml and dockerfiles, and adding model.tar.gz file.
```shell script
python -m automation.enhanced_main_automation \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
    --steps-both
```

#### 2. File updation steps
This runs the initial steps as mentioned above as well as docker building.
```shell script
python -m automation.enhanced_main_automation \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
    --step-6
```

#### 3. Create PR with security analysis and quick checks
This sends a pull request to the deep-learning-container repository, provides analysis of the security vulnerabilities, makes changes to the docker and whitelist files and sends a PR to confirm the changes. 

Sends one final PR to revert toml file.
```shell script
python -m automation.enhanced_main_automation \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
    --create-pr
```

### Post PR Workflow

Remember to turn on VPN via Cisco Client and initialize your credentials using mwinit, before executing the CR files.

#### 1. Send DLContainersInfraCDK CR
```shell script
python -m release.dlc_infra_cr \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
```

#### 2. Release images PR
```shell script
python -m release.autogluon_release_automation \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
    --yaml-only
```

#### 3. Revert release images and update available_images.md PR
```shell script
python -m release.autogluon_release_automation \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
```

#### 4. Send AsimovImageSecurityScan CR
```shell script
python -m release.asimov_scan_cr \
    --current-version 1.3.1 --previous-version 1.3.0 \
    --fork-url https://github.com/Atharva-Rajan-Kale/deep-learning-container.git \
```
