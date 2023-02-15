import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms,models
import boto3
from PIL import Image
import json
import cv2
import numpy as np
import os


def detect_eyes(img):
    
    if os.path.isdir('/opt/ml/model/code'):
        path = '/opt/ml/model/code/haarcascade_frontalface_alt.xml'
    else:
        path = 'haarcascade_frontalface_alt.xml'
    
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = []
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        eye = faces[0:int(h/2), int(w/2):w]
        eyes.append(Image.fromarray(eye))
    return eyes


def get_img(bucket, key):
    client = boto3.client('s3')

    img = client.get_object(Bucket=bucket, Key=key)
    img = img['Body']
    img = Image.open(img)

    return img


def get_bucket_key(path):
    bucket = path.split('/')[2]
    path = '/'.join(path.split('/')[3:])

    return bucket, path


def model_fn(model_dir):
    model = models.resnet152(weights="DEFAULT")
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_dir+'/model_files.pth'))

    return model


def normalize_image(img):
    norm_transform = transforms.Normalize(img.mean([1, 2]), img.std([1, 2]))
    return norm_transform(img)


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
        img_paths = request_body['image_paths']
        img_ids = request_body['image_ids']
    else:
        raise ValueError("This model only supports application/json input")

    transform = transforms.Compose(
        [transforms.Resize(60),
         transforms.ToTensor(),
         ]
    )
    tensor_obj = {}
    for img_path, img_id in zip(img_paths, img_ids):
        bucket, key = get_bucket_key(img_path)
        img = get_img(bucket, key)
        eyes = detect_eyes(img)
        tensor_obj[img_id] = [normalize_image(transform(img)) for img in eyes]

    return tensor_obj


def predict_fn(input_data, model):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for img_id in input_data.keys():
            predictions[img_id] = []
            for img in input_data[img_id]:
                img = img.unsqueeze(0)
                output = model(img)
                _, predicted = torch.max(output, 1)
                predictions[img_id].append(
                    {
                        'prediction': int(predicted.item()),
                        'confidence': torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                        }
                    )

    return json.dumps(predictions)


def output_fn(prediction, content_type):
    prediction = json.loads(prediction)
    classes = ['female', 'male']
    for img_id in prediction.keys():
        for i in range(len(prediction[img_id])):
            prediction[img_id][i]['prediction'] = classes[prediction[img_id][i]['prediction']]
    return json.dumps(prediction)


if __name__ == "__main__":
    input_json = {
        "image_ids": [
            0,
            1,
            2
        ],
        "image_paths": [
            "s3://tomislav-ml-demo/image-data/gender-eye/usage-images/tomislav_cheers.jpg",
            "s3://tomislav-ml-demo/image-data/gender-eye/usage-images/two_ppl_faces_stock.jpg",
            "s3://tomislav-ml-demo/image-data/gender-eye/usage-images/female.jpg"
            
        ]
    }

    model = model_fn("")
    input_data = input_fn(json.dumps(input_json), "application/json")
    prediction = predict_fn(input_data, model)
    out = output_fn(prediction, "application/json")
    print(out)
