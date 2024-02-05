import os  
import glob
import sklearn
from sklearn.model_selection import train_test_split

import PIL 
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
#from torchinfo import summary 

import torch.optim as optim
from IPython.display import Image
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from pathlib import Path
import shutil
import cv2

from torchvision.models import mobilenet_v2

#to check either cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device 

#import dataset from facemask net
'''
!gdown --id 17UdlAXthp-SNQgk1C9dRxe0s20Z1WkLR
!unzip '/content/Correct_Mask_128_128.zip'

!gdown --id 1JPeEeclRZvOMCrSBRxa5OEgTBzoSiHGC
!unzip '/content/thumbnails128x128-20220601T031239Z-001.zip'

!gdown --id 1lKHhsrAkx0qH7YMiNr6hEDKGZN9u3iZG
!unzip '/content/Incorrect_Mask_128_128.zip'

!gdown --id 1Kowgf-mAkiqyFbgGwsTVFWCOyPRoqddP
!unzip '/content/Incorrect_Mask1_128_128.zip'

!gdown --id 16gLX3XqF46g7FnoSmRnctg7ofwY46Kh4
!unzip '/content/Correct_Mask1_128_128.zip'
'''

#move all images from correct_mask1 to correct_mask

file_source ='/content/Correct_Mask1_128_128'
file_destination ='/content/Correct_Mask_128_128'

for file in Path(file_source).glob('*.*'):
    shutil.move(os.path.join(file_source,file),file_destination)


#move all images from incorrect_mask1 to incorrect_mask

file_source ='/content/Incorrect_Mask1_128_128'
file_destination ='/content/Incorrect_Mask_128_128'

for file in Path(file_source).glob('*.*'):
    shutil.move(os.path.join(file_source,file),file_destination)


#initialize random number 
random_seed = 124
np.random.seed(random_seed)

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

#return all file paths for these labels
#train, validation, test

path = '/content'

with_mask = glob.glob(os.path.join(path, "Correct_Mask_128_128",'*.*'))
without_mask = glob.glob(os.path.join(path, 'thumbnails128x128','*.*'))[:54000]
incorrect_mask = glob.glob(os.path.join(path, 'Incorrect_Mask_128_128','*.*'))

images = with_mask + without_mask + incorrect_mask
labels = np.array([1]*len(with_mask)+[0]*len(without_mask)+[2]*len(incorrect_mask))

#split dataset into train, validation and test dataset

images_tv, images_test, y_tv, y_test  = train_test_split(images, labels, shuffle=True, test_size=0.25, random_state=123)
images_train, images_val, y_train, y_val  = train_test_split(images_tv, y_tv, shuffle=True, test_size=0.15, random_state=123)

len(with_mask), len(without_mask), len(incorrect_mask)

#transform for train dataset
train_transforms = transforms.Compose(
    [  
        transforms.Resize((128,128)),
        transforms.ColorJitter(brightness=0.25),  
        transforms.RandomRotation(degrees=45),  
        transforms.RandomHorizontalFlip(p=0.5),  
       # transforms.Grayscale(), 
       # transforms.GaussianBlur(kernel_size=(7,13), sigma=(0.1, 0.2)),
        transforms.ToTensor(), 
        #value of mean and std from the mobilenet documentation 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         
        
    ]
)

#transform for test dataset
test_transforms = transforms.Compose([
  #  transforms.Grayscale(), 
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

class FaceMask_Dataset(Dataset):
    def __init__(self, img_path, img_labels, dataset_type, grayscale=True):
        self.img_path = img_path
        self.img_labels = torch.Tensor(img_labels)
        self.dataset_type = dataset_type
        #self.transforms = img_transforms 

        #transform train dataset
        self.brightness_fct = transforms.ColorJitter(brightness=0.25)
        self.rotate_fct =  transforms.RandomRotation(degrees=45) 
        self.flip_fct = transforms.RandomHorizontalFlip(p=0.5)
        self.normalize_fct = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize_fct = transforms.Resize((128,128))
        self.blur_fct = transforms.GaussianBlur(kernel_size=(7,13), sigma=(0.1, 0.2))
        self.tensor_fct = transforms.ToTensor()

    def __getitem__(self, index):
        # load image
        cur_path = self.img_path[index]
        cur_img = PIL.Image.open(cur_path).convert('RGB')
        #cur_img = self.transforms(cur_img)

        self.resize_fct(cur_img)
        
        if self.dataset_type == 'Train':
          random_num = np.random.randint(1,100)
          if random_num <= 25:

            random_num = np.random.randint(1,100)
            if random_num <= 15:
              cur_img = self.brightness_fct(cur_img)

            random_num = np.random.randint(1,100)
            if random_num <= 15:
              cur_img = self.rotate_fct(cur_img)

            random_num = np.random.randint(1,100)
            if random_num <= 15:
              cur_img = self.flip_fct(cur_img)
            
            random_num = np.random.randint(1,100)
            if random_num <= 15:
              cur_img = self.blur_fct(cur_img)

        cur_img = self.tensor_fct(cur_img)
        cur_img = self.normalize_fct(cur_img)
        

        return cur_img, self.img_labels[index]
    
    def __len__(self):
        return len(self.img_path)


net = mobilenet_v2(pretrained=True)

#freeze the layer
for param in net.parameters():
    param.requires_grad = False

#unfreeze final layer
# net.classifier[1] = torch.nn.Linear(in_features=net.classifier[1].in_features, out_features=3)

net.classifier = torch.nn.Sequential(
          torch.nn.Dropout(p=0.2), 
          torch.nn.Linear(in_features=net.classifier[1].in_features, out_features=256),
          torch.nn.ReLU(),
          torch.nn.Linear(in_features=256, out_features=128),
          torch.nn.ReLU(),
          torch.nn.Linear(in_features=128, out_features=3)
        )

def train_model(model, train_dataset, val_dataset, test_dataset, device, epochs, batch_size, l2):
    model = model.to(device)

    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()  
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)  # pass in the parameters to be updated and learning rate
     #optimizer
    optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
   

    # Training Loop
    print("Training Start:")
    for epoch in range(epochs):
        model.train()  # start to train the model, activate training behavior

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        for i, (images, labels) in enumerate(train_loader):
            #print(images.size())
            # reshape images()
            images = images.to(device)  # reshape: from (128, 1, 28, 28) -> (128, 28 * 28) = (128, 284), move batch to device
            labels = labels.type(torch.LongTensor).to(device)  # move to device

            # forward
            outputs = model(images)  # forward
            pred = outputs.argmax(1)
            cur_train_loss = criterion(outputs, labels)  # lZAss
            cur_train_acc = (pred == labels).sum().item() / batch_size
            
            # backward
            cur_train_loss.backward()   # run back propagation
            optimizer_conv.step()            # optimizer update all model parameters
            optimizer_conv.zero_grad()       # set gradient to zero, avoid gradient accumulating

            # loss
            train_loss += cur_train_loss 
            train_acc += cur_train_acc
        print(epoch)
        # valid
        model.eval()  # start to train the model, activate training behavior
        with torch.no_grad():  # tell pytorch not to update parameters
            for images, labels in val_loader:
                # calculate validation loss
                images = images.to(device)
                labels = labels.type(torch.LongTensor).to(device)
                # outputs = model(images).view(-1)
                outputs = model(images)

                # loss
                cur_valid_loss = criterion(outputs, labels)
                val_loss += cur_valid_loss
                # acc
                #pred = torch.sigmoid(outputs)
                pred = outputs.argmax(1)
                #pred = torch.round(pred)
                val_acc += (pred == labels).sum().item() / batch_size

        # learning schedule step
        #scheduler.step()

        # print training feedback
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)

        print(f"Epoch:{epoch + 1} / {epochs}, lr: {optimizer_conv.param_groups[0]['lr']:.5f} train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")
    
        # update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    test_acc = 0
    return history

# Load the data
train_dataset = FaceMask_Dataset(img_path=images_train, img_labels=y_train, dataset_type='Train')
val_dataset = FaceMask_Dataset(img_path=images_val, img_labels=y_val, dataset_type='Test')
test_dataset = FaceMask_Dataset(img_path=images_test, img_labels=y_test, dataset_type='Test')

# Train the CNN model
#cnn_model = Convnet()
hist1 = train_model(net, train_dataset, val_dataset, test_dataset, device, batch_size=64, epochs=5, l2=0.09)

#save the model
torch.save(net.state_dict(), 'pytorch_mobilenet_6.pth')

#load the model
state_dict = torch.load('/content/pytorch_mobilenet_6.pth')

#test the model with test dataset

device = torch.device('cpu')

PATH1 = "/content/pytorch_mobilenet_5.pth"
model = net.to(device)


def test_model(PATH1, val_dataset, device, test_acc):
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()

    images = []
    labels = []
    probs = []
    test_acc = 0
    with torch.no_grad():
      for (x, y) in test_loader:
          
        x = x.to(device)
        #print(x.size())
        y = y.to(device)
        y_pred = model(x)   
        y_prob = y_pred.argmax(1)
        #y_prob = torch.round(y_prob)

        # images.append(x.to(device))
        labels.append(y.to(device))
        probs.append(y_prob.to(device))
  
        test_acc += (y_prob == y).sum().item()

    # images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    print(f'Test Accuracy: {(test_acc/len(test_loader))}')

    return probs, labels
    
#confusion matrix
def plot_confusion_matrix(labels, pred, classes):
  
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred)
    cm = ConfusionMatrixDisplay(cm)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    ax.xaxis.set_ticklabels(['without mask', 'with mask', 'incorrect mask']); ax.yaxis.set_ticklabels(['without mask', 'with mask', 'incorrect mask']);
    plt.xticks(rotation=2)

#images, labels, probs = test_model(model, test_dataset, device, test_acc=0)
probs, labels = test_model(model, test_dataset, device, test_acc=0)
#pred = probs.argmax(-1)
plot_confusion_matrix(labels, probs, 3)

#test the model with google images
from PIL import Image 
import torch
import face_detection

PATH = "/content/pytorch_mobilenet_5.pth"


maskclasses = ["without mask", "with mask", "incorrect_mask"]

device = torch.device('cpu')

model = mobilenet_v2()
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=3)

model.classifier = torch.nn.Sequential(
          torch.nn.Dropout(p=0.2), 
          torch.nn.Linear(in_features=net.classifier[1].in_features, out_features=256),
          torch.nn.ReLU(),
          torch.nn.Linear(in_features=256, out_features=128),
          torch.nn.ReLU(),
          torch.nn.Linear(in_features=128, out_features=3)
        )

model.load_state_dict(torch.load(PATH, map_location=device))


def classify(model, image_path, maskclasses):
  print()
  model.eval()

  image = PIL.Image.open(image_path).convert('RGB')
  image = np.uint8(image)
  detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.65,nms_iou_threshold=.3)
  bounding_boxes = detector.detect(np.array(image))
  print(image[0,0,:])
  for i in bounding_boxes[:, :4]:
    
    x0, y0, x1, y1 = [int(_) for _ in i]

    face_to_analyze = image[max(11, y0)-10:y0+y1+10, max(11, x0)-10:x0+x1+10]
    face_to_analyze = Image.fromarray(np.uint8(face_to_analyze))
    face_prepared = test_transforms(face_to_analyze)
    image_tensor = face_prepared.unsqueeze(0)
    output = model(image_tensor)

    pred = output.argmax(1) 

    if pred == 0:
      label = "Without Mask"
    elif pred == 1:
      label = "Mask"
    else:
      label = "Incorrect Mask"
  
  # image_tensor = test_transforms(image)
  # print(image)
  # #print(image_tensor)
  # image_tensor = image_tensor.unsqueeze(0)
  # output = model(image_tensor)

  # print(maskclasses[output.argmax(1)])
  # print(output)
  # print(image_tensor[0,0,:])
  # print(image_tensor.size())
  
classify(model, "/content/g2_withmask.JPG", maskclasses)
classify(model, "/content/g1_nomask.jpg", maskclasses)
classify(model, "/content/g1_incorrectmask.JPG", maskclasses)

