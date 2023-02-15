from sagemaker.pytorch import PyTorch
import os

estimator = PyTorch(
    entry_point="../scripts/training_resnet.py",
    role=os.getenv("AWS_SAGEMAKER_ROLE"),
    framework_version="1.12.1",
    py_version="py38",
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={
        'epochs': 20,
        'batch-size': 32,
        'learning-rate': 3e-4,
        }
)

input = 's3://tomislav-ml-demo/image-data/gender-eye/'
estimator.fit(input)


