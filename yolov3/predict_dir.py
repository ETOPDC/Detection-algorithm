'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
from yolo import YOLO
import os


yolo = YOLO()

root = "VOCdevkit/VOC2007/JPEGImages"
save_root = "detect_results"

for a,b,c in os.walk(root):
    for file_i in c:
        file_i_full_path = os.path.join(a,file_i)
        # print(file_i_full_path)
        try:
            image = Image.open(file_i_full_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            # r_image.show()
            r_image.save(os.path.join(save_root,file_i ))

