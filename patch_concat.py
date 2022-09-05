import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

img = Image.open('./ori_img/1005701_apply(moderately).jpg')
w,h = img.size
stepSize = 400
(w_height, w_width) = (400, 400)

#patch 이미지 합치기
ORI_img = Image.new('L', (400 * ((w // 400)+1), 400 * ((h // 400)+1))).convert('RGB')
for i in range(0, h, stepSize):
    for j in range(0, w, stepSize):
        #이미지 불러오기
        im = Image.open('./test_patch_result/1005701_apply(moderately)/test{},{}.jpg'.format(i,j)).convert('RGB')
        
        ORI_img.paste(im, (j, i))

ORI_img.save('./1005701_apply(moderately)_colormap.png')