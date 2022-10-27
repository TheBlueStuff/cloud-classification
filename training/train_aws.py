import sys
import subprocess
import os


REQUIREMENTS_PATH = '/opt/ml/processing/input/requirements'

# subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                        os.path.join(REQUIREMENTS_PATH, "requirements_aws.txt")])

DATA_PATH = '/opt/ml/processing/input/data'
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from torchvision.io import read_image
from torchvision import models
import os
import glob
import random
import pandas as pd
import seaborn as sn
#from cloudnet import *
#from casv3 import *
#from early_stopping import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
from tqdm import tqdm
import cv2
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import wandb
import imgaug
from datetime import datetime

subprocess.run(["wandb", "login", "d579e30bd55604e563481f9625aebcc61c213737"])

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Is cuda available?")
    if torch.cuda.is_available():
        print("Cuda available")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    imgaug.random.seed(seed)

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class CAS(Dataset):
    """CAS Dataset."""
    def __init__(self, transform=None, data_path=None):
        self.data_path = data_path
        self.classes = [
            (0, 'Ac'),
            (1, 'As'),
            (2, 'Cb'),
            (3, 'Ci'),
            (4, 'Cc'),
            (5, 'Cs'),
            (6, 'Cu'),
            (7, 'Ns'),
            (8, 'Sc'),
            (9, 'St')
        ]
        
        self.transform = transform
        self.data = pd.read_csv(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_row = self.data.loc[idx, :]
        try:
            root = self.data_path.split('/')
            root.pop(-1)
            root = '/'.join(root)
            image = cv2.imread(os.path.join(root, image_row['path']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = image_row['n_class']
            if self.transform is not None:
                image = self.transform(image=image)["image"]
        except:
            print(image_row['path'])
            return None, None

        return image, label
    
    @staticmethod
    def get_n_classes():
        return 10


""" Convolutional neural network model """
class CloudNet(nn.Module):
    def __init__(self):
        super(CloudNet, self).__init__()
        self.pool = nn.MaxPool2d(3, stride=2)
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, bias=False)
        self.b1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, bias=False)
        self.b2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 6 * 6, 9216)
        self.fc2 = nn.Linear(9216, 4096)
        self.fc3 = nn.Linear(4096, 5)
        self.dropout = nn.Dropout(0.5)
            
    def forward(self, x):
        x = self.pool(F.relu(self.b1(self.conv1(x))))
        x = self.pool(F.relu(self.b2(self.conv2(x))))
        x = F.relu(self.b3(self.conv3(x)))
        x = self.pool(F.relu(self.b4(self.conv4(x))))
        x = x.view(-1, 256 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
        torch.nn.init.constant_(m.bias, 0)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--architecture', type=str, default='Inceptionv3')
    parser.add_argument('--dataset', type=str, default='CloudAppreciationSociety')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--reduce_factor', type=int, default=10)
    parser.add_argument('--print_epochs', type=int, default=10)
    parser.add_argument('--print_times-per-epoch', type=int, default=20)
    parser.add_argument('--reduce_epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--early_stopping', type=bool, default=False)
    return parser.parse_args()


def main(config):
    set_seed(17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    wandb.login()
    now = datetime.now()
    now_txt = str(now).replace(":","-").replace(".", "-").replace(" ", "_")
    wandb.init(
    # Set the project where this run will be logged
    project="cloud-classification", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{config['architecture']}_{config['dataset']}_{now_txt}", 
    # Track hyperparameters and run metadata
    config=config)

    """ Load pretrained model """
    if config['architecture'] == 'Inceptionv3':
        model = models.inception_v3(pretrained=True)
        in_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_ftrs, CAS.get_n_classes())
        in_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_ftrs, CAS.get_n_classes())
        input_size = 299
    elif config['architecture'] == 'Resnet-50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        in_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_ftrs, CAS.get_n_classes())
        input_size = 224
    elif config['architecture'] == 'CloudNet':
        model = CloudNet()
        # model.load_state_dict(torch.load('./resnet50HBMCD.pth'))
        in_ftrs = model.fc3.in_features
        model.fc3 = nn.Linear(in_ftrs, CAS.get_n_classes())
        input_size = 227

    model = model.to(device)
    if config['architecture'] == 'CloudNet':
        model = model.apply(weights_init)
        
    wandb.watch(model)

    train_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=input_size+25),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=input_size, width=input_size),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    validation_transform = A.Compose(
        [
            A.Resize (height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_data = CAS(transform=train_transform, data_path=os.path.join(DATA_PATH, 'train.csv'))
    validation_data = CAS(transform=validation_transform, data_path=os.path.join(DATA_PATH, 'validation.csv'))
    test_data = CAS(transform=validation_transform, data_path=os.path.join(DATA_PATH, 'test.csv'))

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=40, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=40, shuffle=False, pin_memory=True)

    n_total_steps = len(train_loader)
    n_total_steps

    learning_rate = config["learning_rate"]
    num_epochs = config["epochs"]
    reduce_factor = config["reduce_factor"]
    print_epochs = config["print_epochs"] # 5
    reduce_epochs = config["reduce_epochs"] # 20
    batch_size = config["batch_size"]
    print_times_per_epoch = config["print_times_per_epoch"]
    EARLY_STOPPING = config["early_stopping"]
    running_loss = 0.0
    running_correct = 0
    running_predicted_size = 0
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif config['optimizer'] == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

    classes = train_data.classes
    matrix = np.zeros((len(classes), len(classes)))
    batch_size_validation = 40
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.2)
    model.train()
    stream = tqdm(train_loader)
    for epoch in range(num_epochs):
        labels_f1 = []
        predicted_f1 = []
        for i, (images, labels) in enumerate(stream):
            images = images.to(device)
            labels = labels.to(device)

            if config['architecture'] == 'Inceptionv3':
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_predicted_size += predicted.size(0)
            if (i+1) % (n_total_steps//print_times_per_epoch + 1) == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                running_accuracy = running_correct / running_predicted_size
                running_correct = 0
                running_predicted_size = 0
                running_loss = 0.0
                wandb.log({"loss": loss, "running_accuracy" : running_accuracy})
            
        # Print validation loss and cofusion matrix
        if (epoch+1) % print_epochs == 0  or (epoch+1) == num_epochs:
            # validation the model so far
            model.eval()  # handle drop-out/batch norm layers
            loss_validation = 0
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for _ in range(len(classes))]
            n_class_samples = [0 for _ in range(len(classes))]
            with torch.no_grad():
                for j, (images, labels) in enumerate(validation_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)  # only forward pass - NO gradients!!
                    loss_validation += criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                    labels_f1.append(labels)
                    predicted_f1.append(predicted)
                    for i in range(batch_size_validation):
                        try:
                            label = labels[i]
                            pred = predicted[i]
                            matrix[label][pred] += 1
                            if (label == pred):
                                n_class_correct[label] += 1
                            n_class_samples[label] += 1
                        except:
                            continue
            predicted_f1 = torch.cat(predicted_f1).tolist()
            labels_f1 = torch.cat(labels_f1).tolist()
            f1_macro = f1_score(labels_f1, predicted_f1, average="macro")
            f1_micro = f1_score(labels_f1, predicted_f1, average="micro")
            balanced_accuracy = balanced_accuracy_score(labels_f1, predicted_f1)      
            wandb.log({"f1_macro": f1_macro})
            wandb.log({"f1_micro": f1_micro})
            wandb.log({"balanced_accuracy": balanced_accuracy})
            acc = 100.0 * n_correct / n_samples
            wandb.log({"accuracy": acc})
            # total loss - divide by number of batches
            loss_validation = loss_validation / len(validation_loader)
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss validation: {loss_validation:.4f}')
            wandb.log({"loss_validation": loss_validation})
            wandb.log({"delta_loss": loss_validation - loss})
            
            for i in range(len(classes)):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                matrix[i] = matrix[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')
            
            df_cm = pd.DataFrame(matrix, index = train_data.classes,
                            columns = train_data.classes)
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True)
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            model.train()

        # Learning rate reduction
        if (epoch+1) % reduce_epochs == 0:
            if reduce_factor:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/reduce_factor
                    
        # Early stopping
        if EARLY_STOPPING:
            try:
                early_stopping(loss, loss_validation)
                if early_stopping.early_stop:
                    print("Stopped at epoch:", epoch)
                    break
            except:
                pass


        
    print('Finished Training')
    PATH = os.path.join(wandb.run.dir, "model.pth")
    torch.save(model.state_dict(), PATH)

    # test the model so far
    model.eval()  # handle drop-out/batch norm layers
    loss_test = 0
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(len(classes))]
    n_class_samples = [0 for _ in range(len(classes))]
    matrix = np.zeros((len(classes), len(classes)))
    labels_f1 = []
    predicted_f1 = []
    batch_size_test = 40
    with torch.no_grad():
        for j, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # only forward pass - NO gradients!!
            loss_test += criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            labels_f1.append(labels)
            predicted_f1.append(predicted)
            for i in range(batch_size_test):
                try:
                    label = labels[i]
                    pred = predicted[i]
                    matrix[label][pred] += 1
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
                except:
                    continue
                    
    predicted_f1 = torch.cat(predicted_f1).tolist()
    labels_f1 = torch.cat(labels_f1).tolist()
    avg_accuracy = accuracy_score(labels_f1, predicted_f1)
    balanced_accuracy = balanced_accuracy_score(labels_f1, predicted_f1)
    f1_macro = f1_score(labels_f1, predicted_f1, average="macro")
    f1_micro = f1_score(labels_f1, predicted_f1, average="micro")
    print("average accuracy of the model: ", avg_accuracy)
    print("F1 Macro: ", f1_macro)
    print("F1 Micro: ", f1_micro)
    acc = 100.0 * n_correct / n_samples
    wandb.log({"avg_accuracy": acc})
    wandb.log({"f1_macro_test": f1_macro})
    wandb.log({"f1_micro_test": f1_micro})
    wandb.log({"balanced_accuracy_test": balanced_accuracy})
    # total loss - divide by number of batches
    loss_test = loss_test / len(test_loader)
    wandb.log({"loss_test": loss_test})
    for i in range(len(classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        matrix[i] = matrix[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

    df_cm = pd.DataFrame(matrix, index = train_data.classes,
                    columns = train_data.classes)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    wandb.log({"confusion_matrix_test": wandb.Image(plt)})

    wandb.finish()

if __name__ == '__main__':
    args = read_args()
    config={
    "learning_rate": args.learning_rate,
    "architecture": args.architecture,
    "dataset": args.dataset,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "reduce_factor": args.reduce_factor,
    "print_epochs": args.print_epochs,
    "print_times_per_epoch": args.print_times_per_epoch,
    "reduce_epochs": args.reduce_epochs,
    "optimizer": args.optimizer,
    "early_stopping": args.early_stopping
    }
    main(config)