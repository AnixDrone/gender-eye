import boto3
import time
import argparse


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transform_name', required=True, type=str,
                        help='Name of the BatchTransform job which is going to be created')
    parser.add_argument('-m', '--model_name', required=True, type=str, help='Name of the model which is going to be used')
    parser.add_argument('-i', '--input_uri', required=True, type=str, help='S3 URI of the input data')
    parser.add_argument('-o', '--output_uri', required=True, type=str, help='S3 URI of the output data')
    parser.add_argument('-c', '--instance_count', required=True, type=int, help='Number of instances to be used')
    parser.add_argument('-y', '--instance_type', required=True, type=str, help='Type of the instance to be used')
    
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_command_line_arguments()
    transform_name = args.transform_name
    model_name = args.model_name
    batch_input = args.input_uri
    batch_output = args.output_uri
    instance_count = args.instance_count
    instance_type = args.instance_type
    
    sagemaker_client = boto3.client('sagemaker')
    batch_transformer = sagemaker_client.create_transform_job(
    TransformJobName=f'{transform_name}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}',
    ModelName=model_name,
    BatchStrategy='MultiRecord',
    TransformInput={
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': batch_input
            }
        },
        'ContentType': 'application/json',
        'CompressionType': 'None',
        'SplitType': 'Line'
    },
    TransformOutput={
        'S3OutputPath': batch_output,
        'AssembleWith': 'Line'
    },
    TransformResources={
        'InstanceType': instance_type,
        'InstanceCount': instance_count
    },
    MaxPayloadInMB=100,
    Environment={
        'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'
    },
    ModelClientConfig={
        'InvocationsTimeoutInSeconds': 3600,
        'InvocationsMaxRetries': 1
    }
    )