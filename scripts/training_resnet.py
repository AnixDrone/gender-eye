import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, models
import argparse
import os
import logging
import sys
import torch.optim as optim
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, target_classes, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_classes = target_classes

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
            # custom normalization so that the mean and std are calculated on the fly and are 0 and 1 for every image
            custom_normalize = transforms.Normalize(image.mean([1,2]), image.std([1,2]))
            image = custom_normalize(image)
        label_idx = self.target_classes.index(label)
        return image, label_idx


def initialize_model():
    model = models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    data_dir = args.data_dir
    model_dir = args.model_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ['female', 'male']
    
    transform = transforms.Compose(
        [transforms.Resize(60), 
         transforms.ToTensor(), 
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )
    train_dataset = CustomImageDataset(
        data_dir+'/train.csv', data_dir, classes, transform=transform)
    test_dataset = CustomImageDataset(
        data_dir+'/test.csv', data_dir, classes, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    net = initialize_model().to(device) # Net().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            inputs, labels = data.to(device), target.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logger.info(
            f'Epoch {epoch+1} - Training loss: {running_loss/len(train_loader)}')

        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            logger.info('Accuracy of the network on test images: %d %%' % (
                100 * correct / total))

    torch.save(net.state_dict(), os.path.join(model_dir, "model.pth"))
