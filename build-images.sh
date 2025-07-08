#!/bin/bash

# Set your AWS account ID and region
ACCOUNT_ID=644385875248
REGION="us-east-1"

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Create ECR repositories if they don't exist
aws ecr create-repository --repository-name autogluon-automation --region $REGION || true
aws ecr create-repository --repository-name autogluon-test --region $REGION || true

# Build automation image
docker build -f Dockerfile.automation -t autogluon-automation .
docker tag autogluon-automation:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/autogluon-automation:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/autogluon-automation:latest

# Build test image
docker build -f Dockerfile.testing -t autogluon-test .
docker tag autogluon-test:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/autogluon-test:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/autogluon-test:latest

echo "Images built and pushed successfully!"