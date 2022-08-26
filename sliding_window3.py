import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance 

Image.MAX_IMAGE_PIXELS = None

# img = cv2.imread("./ori_img/1004490_mini.jpg", cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.open('./ori_img/1004589_apply(normal).jpg').convert('RGB')

w,h = img.size
stepSize = 400
(w_height, w_width) = (400, 400)
for x in range(0, h, stepSize):
    for y in range(0, w, stepSize):
        #img[h:h + w_height, w:w + w_width] = patch #이렇게 만들어봐라
        #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = img.crop((y, x, y+w_width, x+w_height))
        # cropped = cropped.convert('RGB')
        # enhancer = ImageEnhance.Brightness(cropped)
        # cropped = enhancer.enhance(2).convert('RGB')
        #cv2.imwrite('./test_patch/test{},{}.jpg'.format(h+200,w), cropped)
        cropped.save('./test_patch/1004589_apply(normal)/test{},{}.jpg'.format(x,y))





'''
img = cv2.imread("./ori_img/1004490.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,_ = img.shape
stepSize = 400

(w_height, w_width) = (400, 400)
for x in range(0, h - w_height, stepSize):
    for y in range(0, w - w_width, stepSize):
        cropped = img[x:x + w_height, y:y + w_width]
        #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # cropped = img.crop((w, h, w+w_width, h+w_height))
        # cropped = cropped.convert('RGB')

        cv2.imwrite('./test_patch/test{},{}.png'.format(x,y), cropped)
        # cropped.save('./test_patch/test{},{}.png'.format(h+200,w))
'''