import os
import cv2
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from matplotlib import pyplot as plt
from dataset import ColorectalDataset

if __name__ == '__main__':
    
    DIR_TEST = "./test/"

    classes = os.listdir(DIR_TEST)
    test_imgs = []

    for _class in classes:
            
        for img in os.listdir(DIR_TEST + _class):
            test_imgs.append(DIR_TEST + _class + "/" + img)

    class_to_int = {'normal':0, 'adenoma':1, 'well':2, 'moderately':3, 'poorly':4}
    print(class_to_int)    #클래스 별 라벨값 입력

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()


    model = torch.load('./weights/wt/best_model_epoch73.pth')
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()  

    Batch_Size = 1

    def get_transform():
        return T.Compose([T.ToTensor()])  

    def calc_accuracy(true,pred):
        pred = F.softmax(pred, dim = 1)
        true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
        acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
        acc = float((100 * acc.sum()) / len(acc))
        return round(acc, 4)

    test_dataset = ColorectalDataset(test_imgs, class_to_int, get_transform())

    test_data_loader = DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size = Batch_Size,
        num_workers = 4,
    )

    test_loss = []
    test_accuracy = []


    #test
    for images, labels in test_data_loader:
        
        images = images.to(device)
        labels = labels.to(device)
        
        preds = model(images)
        
        acc = calc_accuracy(labels.cpu(), preds.cpu())
        
        loss = criterion(preds, labels)
        
        loss_value = loss.item()
        test_loss.append(loss_value)
        test_accuracy.append(acc)
        
    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)


    print("Test Loss = {}".format(round(test_loss, 4)))
    print("Test Accuracy = {} % \n".format(test_accuracy))











# def testAccuracy():
#     model.eval()
#     accuracy = 0.0
#     total = 0.0
    
#     with torch.no_grad():
#         for data in test_data_loader:
#             images, labels = data
#             # run the model on the test set to predict labels
#             outputs = model(images)
#             # the label with the highest energy will be our prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             accuracy += (predicted == labels).sum().item()
    
#     # compute the accuracy over all test images
#     accuracy = (100 * accuracy / total)
#     return(accuracy)

# model = models.resnet101()
# model.load_state_dict(torch.load('./weights/best_model_epoch15.pth'))
# model.eval()

# def testClassess():
#     Batch_Size = 4
#     class_correct = list(0. for i in range(labels))
#     class_total = list(0. for i in range(labels))
#     with torch.no_grad():
#         for data in test_data_loader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(Batch_Size):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1

#     for i in range(labels):
#         print('Accuracy of %5s : %2d %%' % (
#             labels[i], 100 * class_correct[i] / class_total[i]))