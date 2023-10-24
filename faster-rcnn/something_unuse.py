
from PIL import Image


import colorsys
import copy
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from nets.frcnn import FasterRCNN
from utils.utils import DecodeBox, get_new_img_size, loc2bbox, nms

img = "img\inclusion_5.jpg"

image = Image.open(img)

# 画框框
# label = '{} {:.2f}'.format(predicted_class, score)
label1 = '{} {:.2f}'.format("inclusion", 0.96)
label2 = '{} {:.2f}'.format("inclusion", 0.99)
label3 = '{} {:.2f}'.format("inclusion", 0.95)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
label_size1 = draw.textsize(label1, font)
label_size2 = draw.textsize(label2, font)
label_size3 = draw.textsize(label3, font)

label1 = label1.encode('utf-8')
label2 = label2.encode('utf-8')
label3 = label3.encode('utf-8')

left1, top1, right1, bottom1 = 141 ,89,166,186
left2, top2, right2, bottom2 = 143 ,11,158,39
left3, top3, right3, bottom3 = 140 ,49,168,90


print(label1, top1, left1, bottom1, right1)

if top1 - label_size1[1] >= 0:
    text_origin1 = np.array([left1, top1 - label_size1[1]])
else:
    text_origin1 = np.array([left1, top1 + 1])


if top2 - label_size2[1] >= 0:
    text_origin2 = np.array([left2, top2 - label_size2[1]])
else:
    text_origin2 = np.array([left2, top2 + 1])

if top3 - label_size3[1] >= 0:
    text_origin3 = np.array([left3, top3 - label_size3[1]])
else:
    text_origin3 = np.array([left3, top3 + 1])


hsv_tuples = [
    (x / 2, 1., 1.)
        for x in range(2)
]  # 获得hsv格式的不同色度

colors = list(
    map(
        lambda x: colorsys.hsv_to_rgb(*x),
        hsv_tuples
    )
)  # 获得rgb格式的不同颜色

colors = list(
    map(
        lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors
    )
)  # 通过hsv格式来调整不同类别对应边框的色度


draw.rectangle([left1, top1 , right1 , bottom1 ],outline=colors[0],width=1)
draw.rectangle([left2, top2 , right2 , bottom2 ],outline=colors[0],width=1)
draw.rectangle([left3, top3 , right3 , bottom3 ],outline=colors[0],width=1)

draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)],outline=colors[0],width=1)
draw.text(text_origin1, str(label1, 'UTF-8'), fill=(1, 0, 0), font=font)

draw.rectangle([tuple(text_origin2), tuple(text_origin2 + label_size2)],outline=colors[0],width=1)
draw.text(text_origin2, str(label2, 'UTF-8'), fill=(1, 0, 0), font=font)

draw.rectangle([tuple(text_origin3), tuple(text_origin3 + label_size3)],outline=colors[0],width=1)
draw.text(text_origin3, str(label3, 'UTF-8'), fill=(1, 0, 0), font=font)

image.show()