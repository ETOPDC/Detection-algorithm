from torch.utils.data import Dataset
from PIL import Image

class MyDataSet(Dataset):
    def __init__(self, dataset_type, transform=None, update_dataset=False):
        """
        dataset_type: ['train', 'test']
        """

        dataset_path = '数据集路径'

        if update_dataset:
            make_txt_file(dataset_path)  # update datalist

        self.transform = transform
        self.sample_list = list()
        self.dataset_type = dataset_type
        f = open(dataset_path + self.dataset_type + '/datalist.txt')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()

    def __getitem__(self, index):
        item = self.sample_list[index]
        # img = cv2.imread(item.split(' _')[0])
        img = Image.open(item.split(' _')[0])
        if self.transform is not None:
            img = self.transform(img)
        label = int(item.split(' _')[-1])
        return img, label

    def __len__(self):
        return len(self.sample_list)



def make_txt_file(dataset_path):
    msg="文件1路径 _类别+检测框\r\n文件2路径 类别+检测框\r\n文件3路径 类别+检测框\r\n文件4路径 类别+检测框\r\n"
    fullpath=dataset_path+"datalist.txt"
    file=open(fullpath,'w')
    file.write(msg)