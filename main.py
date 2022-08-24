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


DIR_TRAIN = "./train/"
DIR_VALID = "./valid/"
DIR_TEST = "./test/"



def calc_accuracy(true,pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)


    

if __name__ == '__main__':
    ### Exploring Dataset  데이터셋 탐색

    classes = os.listdir(DIR_TRAIN)     #train dataset list
    print("Total Classes: ",len(classes))    #총 class 개수

    #Counting total train, valid & test images

    train_count = 0     
    valid_count = 0
    test_count = 0
    Batch_Size = 32
    
    for _class in classes:
        train_count += len(os.listdir(DIR_TRAIN + _class))
        valid_count += len(os.listdir(DIR_VALID + _class))
        test_count += len(os.listdir(DIR_TEST + _class))

    print("Total train images: ",train_count)
    print("Total valid images: ",valid_count)
    print("Total test images: ",test_count)
    
    
    ### Creating a list of all images : DIR_TRAIN/class_folder/img.jpg - FOR METHOD 2 of data loading
    #   A dict for mapping class labels to index    #클래스 레이블을 index에 매핑

    train_imgs = []
    valid_imgs = []
    test_imgs = []

    for _class in classes:
        
        for img in os.listdir(DIR_TRAIN + _class):
            train_imgs.append(DIR_TRAIN + _class + "/" + img)    #train_imgs에 이미지 경로 추가

        for img in os.listdir(DIR_VALID + _class):
            valid_imgs.append(DIR_VALID + _class + "/" + img)
            
        for img in os.listdir(DIR_TEST + _class):
            test_imgs.append(DIR_TEST + _class + "/" + img)

    class_to_int = {'normal':0, 'adenoma':1, 'well':2, 'moderately':3, 'poorly':4}
    print(class_to_int)    #클래스 별 라벨값 입력




    ### Loading Classification Dataset - FOR METHOD 2: For multi-class data, by inheriting Dataset class
    # 다중 클래스 데이터의 경우 Dataset 클래스를 상속하여 만듬


    def get_transform():
        return T.Compose([T.ToTensor()])    #배열구조 H x W x C -> C x H x W로 바뀜


    #여기서 dataset class 만들기
    

    ### Loading Classification Dataset
    """
    # Method 1: For multi-class data directly from folders using ImageFolder    # 방법 1: ImageFolder를 사용하여 폴더에서 직접 다중 클래스 데이터의 경우
    train_dataset = ImageFolder(root = DIR_TRAIN, transform = T.ToTensor())
    valid_dataset = ImageFolder(root = DIR_VALID, transform = T.ToTensor())
    test_dataset = ImageFolder(root = DIR_TEST, transform = T.ToTensor())
    """
    
    # Method 2: Using Dataset Class
    train_dataset = ColorectalDataset(train_imgs, class_to_int, get_transform())
    valid_dataset = ColorectalDataset(valid_imgs, class_to_int, get_transform())
    test_dataset = ColorectalDataset(test_imgs, class_to_int, get_transform())

    #Data Loader  -  using Sampler (YT Video)               #Data Loader - 샘플러 사용(YT Video)
    train_random_sampler = RandomSampler(train_dataset)     #데이터 무작위로 섞기위해 사용
    valid_random_sampler = RandomSampler(valid_dataset)     #mini-batch(표본) 내부 구성이 다양할수록 전체 dataset(모집단)를 잘 대표하기 때문에 주로 RandomSampler를 사용한다.
    test_random_sampler = RandomSampler(test_dataset)

    #Shuffle Argument is mutually exclusive with Sampler!    #Shuffle Argument는 Sampler와 상호 배타적입니다!
    train_data_loader = DataLoader(
        dataset = train_dataset,
        batch_size = Batch_Size,
        sampler = train_random_sampler,
        num_workers = 4,
    )

    valid_data_loader = DataLoader(
        dataset = valid_dataset,
        batch_size = Batch_Size,
        sampler = valid_random_sampler,
        num_workers = 4,
    )


    
    # # Visualize one training batch    ## 하나의 훈련 배치 시각화
    # for images, labels in train_data_loader:
    #     fig, ax = plt.subplots(figsize = (10, 10))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.imshow(make_grid(images, 4).permute(1,2,0))
    #     break
    
    
    
    ### Define model
    model = models.resnet50(pretrained = True)     #모델 resnet50
    ### Modifying last few layers and no of classes
    # NOTE: cross_entropy loss takes unnormalized op (logits), then function itself applies softmax and calculates loss, so no need to include softmax here
    ### 클래스가 없는 마지막 몇 개의 레이어 수정
    # 참고: cross_entropy loss는 비정규화 연산(logits)을 취한 다음 함수 자체에서 softmax를 적용하고 손실을 계산하므로 여기에 softmax를 포함할 필요가 없습니다.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    
    #서버로 돌릴때
    model = nn.DataParallel(model)
    
    
    ### Get device

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    model.to(device)

    ### Training Details    #training 정보

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = [80,120,160,180], gamma = 0.1)    #일정한 Step 마다 learning rate에 gamma를 곱해주는 방식
    criterion = nn.CrossEntropyLoss()    #다중분류에 사용

    train_loss = []
    train_accuracy = []

    val_loss = []
    val_accuracy = []
    
    test_loss = []
    test_accuracy = []

    epochs = 200
    min_val_loss = 99999
    
    
    ### Training Code

    for epoch in range(epochs):
        
        start = time.time()
        
        #Epoch Loss & Accuracy
        train_epoch_loss = []
        train_epoch_accuracy = []
        _iter = 1    #iter는 반복을 끝낼 값을 지정하면 특정 값이 나올 때 반복을 끝냄
        
        #Val Loss & Accuracy
        val_epoch_loss = []
        val_epoch_accuracy = []
        
        test_epoch_loss = []
        test_epoch_accuracy = []
        
        
        # Training
        for images, labels in train_data_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            #Reset Grads
            optimizer.zero_grad()
            
            #Forward ->
            preds = model(images)
            
            #Calculate Accuracy
            acc = calc_accuracy(labels.cpu(), preds.cpu())
            
            #Calculate Loss & Backward, Update Weights (Step)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            #Append loss & acc
            loss_value = loss.item()
            train_epoch_loss.append(loss_value)
            train_epoch_accuracy.append(acc)
            
            if _iter % 50 == 0:
                print("> Iteration {} < ".format(_iter))
                print("Iter Loss = {}".format(round(loss_value, 4)))
                print("Iter Accuracy = {} % \n".format(acc))
            
            _iter += 1
        
        #Validation
        for images, labels in valid_data_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            #Forward ->
            preds = model(images)
            
            #Calculate Accuracy
            acc = calc_accuracy(labels.cpu(), preds.cpu())
            
            #Calculate Loss
            loss = criterion(preds, labels)    #criterion = nn.CrossEntropyLoss()
            
            #Append loss & acc
            loss_value = loss.item()
            val_epoch_loss.append(loss_value)
            val_epoch_accuracy.append(acc)
                    
                
        
        train_epoch_loss = np.mean(train_epoch_loss)
        train_epoch_accuracy = np.mean(train_epoch_accuracy)
        
        val_epoch_loss = np.mean(val_epoch_loss)
        val_epoch_accuracy = np.mean(val_epoch_accuracy)
        
        end = time.time()
        
        #그래프 그릴때 필요할듯
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        
        
        
        val_mean_loss = np.mean(val_loss)
        val_mean_accuracy = np.mean(val_accuracy)
        
        
        if min_val_loss > val_epoch_loss:
            min_val_loss = val_epoch_loss
            print("save model...")
            torch.save(model, './weights/best_model_epoch{}.pth'.format(epoch))
        
        
        #Print Epoch Statistics
        print("** Epoch {} ** - Epoch Time {}".format(epoch, int(end-start)))
        print("Train Loss = {}".format(round(train_epoch_loss, 4)))
        print("Train Accuracy = {} % \n".format(train_epoch_accuracy))
        print("Val Loss = {}".format(round(val_epoch_loss, 4)))
        print("Val Accuracy = {} % \n".format(val_epoch_accuracy))
        
    plt.figure(figsize =(10,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch,train_loss)
    plt.plot(epoch,val_loss)
    plt.savefig('./train&val_loss_graph.png')
    
    plt.figure(figsize =(10,5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epoch,train_accuracy)
    plt.plot(epoch,val_accuracy)
    plt.savefig('./train&val_accuracy_graph.png')