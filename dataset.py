from configparser import Interpolation
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

class ColorectalDataset(Dataset):
    
    def __init__(self, imgs_list, class_to_int, transforms = None):
        
        super().__init__()
        self.imgs_list = imgs_list
        self.class_to_int = class_to_int
        self.transforms = transforms
        
        
    def __getitem__(self, index):
    
        image_path = self.imgs_list[index]      #이미지 경로 
        #Reading image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)    #이미지 읽기(이미지 파일은 Color로 읽어 들임)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)    #BGR 이미지를 RGB 이미지로 변환하기 + float32의 dtype으로 변경
        # image = cv2.resize(image,(400,400))
        image /= 255.0    #픽셀 값을 0~255사이에서 0~1범위로 정규화 하는 것
        
        #Retriving class label    #클래스 레이블 검색
        label = image_path.split("/")[-2]    #경로에서 라벨값 뽑기
        label = self.class_to_int[label]    #라벨값을 0~4 숫자로 바꿔주기    
        
        #Applying transforms on image    #이미지에 변형 적용
        if self.transforms:
            image = self.transforms(image)    #이미지 바뀐 것 적용 
        
        return image, label
        
        
        
    def __len__(self):
        return len(self.imgs_list)    #궁금
    
    
    
#TestDataset    
class ColorectalTestDataset(Dataset):
    
    def __init__(self, imgs_list, transforms = None):
        
        super().__init__()
        self.imgs_list = imgs_list
        self.transforms = transforms
        
        
    def __getitem__(self, index):
    
        image_path = self.imgs_list[index]      #이미지 경로 
        #Reading image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)    #이미지 읽기(이미지 파일은 Color로 읽어 들임)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)    #BGR 이미지를 RGB 이미지로 변환하기 + float32의 dtype으로 변경
        # image = cv2.resize(image,(400,400))
        image /= 255.0    #픽셀 값을 0~255사이에서 0~1범위로 정규화 하는 것
        
        #Retriving class label    #클래스 레이블 검색
        #label = image_path.split("/")[-2]    #경로에서 라벨값 뽑기
        #label = self.class_to_int[label]    #라벨값을 0~4 숫자로 바꿔주기    
        
        #Applying transforms on image    #이미지에 변형 적용
        if self.transforms:
            image = self.transforms(image)    #이미지 바뀐 것 적용 
        
        return image, image_path
        
        
        
    def __len__(self):
        return len(self.imgs_list)    #궁금
    
    