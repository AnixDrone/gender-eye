from amazon_s3.s3_utils import upload_json_data_to_s3
import json

def create_input_for_batch_transform(img_paths):
    bt_input = {
        'image_ids':[i for i in range(len(img_paths))],
        'image_paths':img_paths
    }
    upload_json_data_to_s3('tomislav-ml-demo', 'batch_transform/input/testing.json', bt_input)
    return 's3://tomislav-ml-demo/batch_transform/input/testing.json'

def get_output_transform_output():
    return 's3://tomislav-ml-demo/batch_transform/output/testing.json'

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
            "s3://tomislav-ml-demo/image-data/gender-eye/usage-images/female.jpg",
            "s3://tomislav-ml-demo/image-data/gender-eye/usage-images/trip-to-the-mall-v0-qfyvvg40r2ia1.png"
            
        ]
    }
    path = create_input_for_batch_transform(input_json['image_paths'])
    print(path)