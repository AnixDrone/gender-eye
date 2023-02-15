#Setup
import boto3
import os

region = boto3.Session().region_name
client = boto3.client("sagemaker", region_name=region)

#Role to give SageMaker permission to access AWS services.
sagemaker_role = os.getenv("AWS_SAGEMAKER_ROLE")

response = client.create_endpoint_config(
   EndpointConfigName="gender-eye-endpoint-config",
   ProductionVariants=[
        {
            "ModelName": "gender-eye-13-02-2023-17-06-00",
            "VariantName": "AllTraffic",
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 1
            }
        } 
    ]
)
response = client.create_endpoint(
    EndpointName="gender-eye-endpoint",
    EndpointConfigName='gender-eye-endpoint-config'
)