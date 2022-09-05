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
from dataset import ColorectalTestDataset

if __name__ == '__main__':
    
    DIR_TEST = "./test_patch/1005701_apply(moderately)/"

    classes = os.listdir(DIR_TEST)
    test_imgs = []

    for img_name in classes:
        test_imgs.append(DIR_TEST + img_name)

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

    test_dataset = ColorectalTestDataset(test_imgs, get_transform())

    test_data_loader = DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size = Batch_Size,
        num_workers = 4,
    )

    test_loss = []
    test_accuracy = []
    
    class_to_color = [(255,0,0), (0,255,0), (0,0,255), (0,120,255), (0,250,255)] 
    # 파, 초 , 빨 , 주 , 노  normal, adenoma, well, moderately, poorly
    #test
    for images, image_path in test_data_loader:
        images = images.to(device)
        output = model(images)       
        softmax = nn.Softmax(dim=1)
        softmax_output = softmax(output).detach().cpu().numpy()
        #print(np.argmax(softmax_output, axis=1))
        #print(softmax_output[0][np.argmax(softmax_output, axis=1)])
        #print(image_path)
        img_colormap = np.ones(shape=(400,400,3), dtype=np.int16)
        # print(img_colormap[0, :, :].shape)
        if np.argmax(softmax_output, axis=1) == 0:
            img_colormap[:, :, 0] = img_colormap[:, :, 0] * class_to_color[0][0]
            img_colormap[:, :, 1] = img_colormap[:, :, 1] * class_to_color[0][1]
            img_colormap[:, :, 2] = img_colormap[:, :, 2] * class_to_color[0][2]
        elif np.argmax(softmax_output, axis=1) == 1:
            img_colormap[:, :, 0] = img_colormap[:, :, 0] * class_to_color[1][0]
            img_colormap[:, :, 1] = img_colormap[:, :, 1] * class_to_color[1][1]
            img_colormap[:, :, 2] = img_colormap[:, :, 2] * class_to_color[1][2]
        elif np.argmax(softmax_output, axis=1) == 2:
            img_colormap[:, :, 0] = img_colormap[:, :, 0] * class_to_color[2][0]
            img_colormap[:, :, 1] = img_colormap[:, :, 1] * class_to_color[2][1]
            img_colormap[:, :, 2] = img_colormap[:, :, 2] * class_to_color[2][2]
        elif np.argmax(softmax_output, axis=1) == 3:
            img_colormap[:, :, 0] = img_colormap[:, :, 0] * class_to_color[3][0]
            img_colormap[:, :, 1] = img_colormap[:, :, 1] * class_to_color[3][1]
            img_colormap[:, :, 2] = img_colormap[:, :, 2] * class_to_color[3][2]
        elif np.argmax(softmax_output, axis=1) == 4:
            img_colormap[:, :, 0] = img_colormap[:, :, 0] * class_to_color[4][0]
            img_colormap[:, :, 1] = img_colormap[:, :, 1] * class_to_color[4][1]
            img_colormap[:, :, 2] = img_colormap[:, :, 2] * class_to_color[4][2]
            
        cv2.imwrite('./test_patch_result/1005701_apply(moderately)/'+image_path[0].split('/')[-1], img_colormap)








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