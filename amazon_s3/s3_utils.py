import boto3
from PIL import Image
import logging
import json

def get_session(session_name):
    return boto3.Session(profile_name=session_name)

def get_s3_client(profile_name=None):
    if profile_name is None:
        return boto3.client('s3')
    return get_session(profile_name).client('s3')

def get_img(bucket, key):
    client = get_s3_client()

    img = client.get_object(Bucket=bucket, Key=key)
    img = img['Body']
    img = Image.open(img)

    return img

def upload_json_data_to_s3(bucket, key, data ):
    s3 = get_s3_client()
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=(json.dumps(data).encode('UTF-8'))
        )
    except Exception as e:
        logging.error(f'Could not upload data to bucket {bucket}, key {key}.\n{e}')
        raise e

def get_json_from_s3(bucket, key, profile_name=None):
    s3 = get_s3_client(profile_name)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'])
    except Exception as e:
        logging.error(f'Could not get object from bucket {bucket}, key {key}.\n{e}')
        raise e

def upload_file(bucket,key,filepath,profile_name=None):
    s3 = get_s3_client(profile_name)
    try:
        s3.upload_file(filepath, bucket, key)
    except Exception as e:
        logging.error(f'Could not upload file {filepath} to bucket {bucket}, key {key}.\n{e}')
        raise e
    