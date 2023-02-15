import boto3
from sagemaker.pytorch.model import PyTorchModel
import argparse
import time


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', required=True, type=str,
                        help='Name of the SageMaker model which is going to be created')
    parser.add_argument('-r', '--role_name', required=True, type=str, help='Name of the Amazon SageMaker service role')
    parser.add_argument('-model', '--model_uri', required=True, type=str, help="S3 URI of the model data")
    parser.add_argument('-e', '--entry_point', required=False, type=str, help='Script for entry point')
    return parser.parse_args()


def get_role_arn_from_name(role_name):
    iam_client = boto3.client('iam')
    role = iam_client.get_role(RoleName=role_name)['Role']['Arn']
    return role


def create_pytorch_model(args):
    model_name = args.model_name
    model_data = args.model_uri
    role_arn = get_role_arn_from_name(args.role_name)

    model = PyTorchModel(
        name=model_name + '-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()),
        model_data=model_data,
        entry_point=args.entry_point,
        image_uri="763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38",
        #framework_version="1.12.1",
        #py_version="py38",
        role=role_arn,
    )

    model.create()


if __name__ == '__main__':
    args = parse_command_line_arguments()
    create_pytorch_model(args)