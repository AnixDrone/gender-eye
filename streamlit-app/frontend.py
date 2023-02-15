import streamlit as st
import json
import boto3
import uuid
from PIL import Image
import io


def upload_img_to_s3(img, bucket, key):
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=img)
    return f's3://{bucket}/{key}'

def invoke_endpoint(image_path,image_id):
    smrt = boto3.Session().client(service_name='runtime.sagemaker')
    endpoint_name = "gender-eye-endpoint"
    body = {
        'image_ids': [str(image_id)],
        'image_paths': [image_path]
    }

    response = smrt.invoke_endpoint(EndpointName=endpoint_name,
                                    ContentType='application/json',
                                    Body=json.dumps(body))
    
    response_json = json.loads(response['Body'].read())
    
    return response_json       

st.title("Gender Eye Classifier")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # img = Image.open(img_file_buffer)
    # img = img.resize((56, 56))
    # print(img.size)
    image_id = uuid.uuid4()
    # buf = io.BytesIO()
    # img.save(buf, format='PNG')
    img_path = upload_img_to_s3(img_file_buffer,'tomislav-ml-demo',f'image-data/gender-eye/usage-images/{image_id}.png')
    endpoint_response = invoke_endpoint(img_path,image_id)
    print(endpoint_response)
    st.write(endpoint_response)