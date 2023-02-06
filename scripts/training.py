import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import argparse
import os
import logging
import sys
import torch.optim as optim
import boto3
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, target_classes,transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_classes = target_classes

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        bucket = img_path.split('/')[2]
        key = '/'.join(img_path.split('/')[3:])
        image = get_img(bucket,key)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        label_idx = self.target_classes.index(label)
        return image, label_idx

def save_model(model,model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), path)

def get_img(bucket,key):
    client = boto3.client('s3')
    
    img = client.get_object(Bucket=bucket, Key=key)
    img = img['Body']
    img = Image.open(img)
    
    return img

def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--img-folder', type=str, default='data')

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    batch_size=args.batch_size
    learning_rate=args.learning_rate
    epochs=args.epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes=['female','male']

    transform=transforms.Compose(
        [   transforms.Resize(60)
            ,transforms.ToTensor()
            ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = CustomImageDataset(args.train+'/train.csv',f'../data',classes,transform=transform)
    test_dataset = CustomImageDataset(args.train+'/test.csv',f'../data',classes,transform=transform)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    net = Net().to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        running_loss=0.0
        for data,target in train_loader,0:
            inputs,labels=data.to(device),target.to(device)

            optimizer.zero_grad()

            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}')
            
        with torch.no_grad():
            correct=0
            total=0
            for data in test_loader:
                images,labels=data[0].to(device),data[1].to(device)
                outputs=net(images)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
            print('Accuracy of the network on test images: %d %%' % (
                100 * correct / total)) 

    torch.save(net.state_dict(), os.path.join(args.model_dir, "model.pth"))