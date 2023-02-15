from sagemaker import image_uris
import boto3

if __name__ == "__main__":
    
    # Specify your AWS Region
    aws_region=boto3.Session().region_name

    
    container = image_uris.retrieve(region=aws_region, 
                                   framework='pytorch',
                                   version='1.12.1', 
                                   image_scope='inference',
                                   instance_type='ml.m5.xlarge')
    
    sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

    sagemaker_role= "arn:aws:iam::<account>:role/*"

    s3_bucket = '<your-bucket-name>' # Provide the name of your S3 bucket
    bucket_prefix='saved_models'
    model_s3_key = f"{bucket_prefix}/demo-xgboost-model.tar.gz"

    #Specify S3 bucket w/ model
    model_url = f"s3://{s3_bucket}/{model_s3_key}"

    # Specify an AWS container image. 
    container = image_uris.retrieve(region=aws_region, 
                                   framework='pytorch',
                                   version='1.12.1', 
                                   image_scope='inference',
                                   instance_type='ml.m5.xlarge')

    model_name = '<The_name_of_the_model>'

    #Create model
    create_model_response = sagemaker_client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = sagemaker_role,
        PrimaryContainer = {
            'Image': container,
            'ModelDataUrl': model_url,
        })