import boto3
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--endpoint_name', required=True, type=str, help='Name of the endpoint which is going to be invoked')
    parser.add_argument('-i', '--input_file', required=True, type=str, help='Path to the json input file')
    
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    smrt = boto3.Session().client(service_name='runtime.sagemaker')
    endpoint_name = args.endpoint_name
    with open(args.input_file, 'r') as f:
        body = json.load(f)
    print(f'The input file is: {body}')

    response = smrt.invoke_endpoint(EndpointName=endpoint_name,
                                    ContentType='application/json',
                                    Body=json.dumps(body))
    
    response_json = json.loads(response['Body'].read())
    
    print(f'The response is: {response_json}')