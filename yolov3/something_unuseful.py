import torch
import numpy as np

# 13*13
batch_size=2
num_anchor=3
# grid_x = torch.linspace(0, 12, 13).repeat(13,1).repeat(batch_size*num_anchor,1,1).view([batch_size,num_anchor,13,13])
# grid_y = torch.linspace(0,12,13).repeat(13,1).t().repeat(batch_size*num_anchor,1,1).view([batch_size,num_anchor,13,13])
#
# print(grid_x.size())
# print(grid_x)
#
# print(grid_y.size())
# print(grid_y)

# FloatTensor = torch.cuda.FloatTensor
# _scale = torch.Tensor([13,32]*2).type(FloatTensor)
#print(_scale)

# image_pred=torch.rand([10,5])
# print(image_pred)
# conf_mask = (image_pred[:,4]>=0.5).squeeze()
# print(conf_mask)

# a=np.array([1,2,3])
# b=np.array([4,5,6])
#
# s=np.stack((a,b),axis=1)
# print(a[0::4])



# 权重文件相关操作
from nets.yolo3 import YoloBody
from utils.config import Config

yolo = YoloBody(Config)
state_dict = torch.load("model_data/yolo_weights.pth")
yolo.load_state_dict(state_dict)
#增加层和参数
state_dict["newlayer"]=torch.FloatTensor([1.3,1.5])




#
# for i, p in enumerate(list(yolo.state_dict())):
#     print(i, ":", p)
# #
print("!!!!!!!!!!!!!!!!!!!!!!!!!")
i=0
# for k, v in yolo.state_dict().items():
#     print(i,k)
#     i=i+1
# #
# print("&&&&&&&&&&&&&&&&&&&&&&&&")
i=0
for name,param in yolo.named_parameters():
  print(i,name)
  i=i+1


#0-221    3   --> 0-224
#200 198 199

# for i, p in enumerate(yolo.parameters()):
#     # sml:414-437  smh:462-469
#     if i < 199:
#         p.requires_grad = False
#     elif i > 198 and i < 201:
#         p.requires_grad = True
#     else:
#         p.requires_grad = False

#冻结某些层的方案。。。。。。。。。。。
from collections.abc import Iterable
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)


# 冻结第一层
freeze_by_idxs(yolo, 0)
# 冻结第一、二层
freeze_by_idxs(yolo, [0, 1])
# 冻结倒数第一层
freeze_by_idxs(yolo, -1)
# 解冻第一层
unfreeze_by_idxs(yolo, 0)
# 解冻倒数第一层
unfreeze_by_idxs(yolo, -1)
# 冻结 em层
freeze_by_names(yolo, 'em')
# 冻结 fc1, fc3层
freeze_by_names(yolo, ('fc1', 'fc3'))
# 解冻em, fc1, fc3层
unfreeze_by_names(yolo, ('em', 'fc1', 'fc3'))