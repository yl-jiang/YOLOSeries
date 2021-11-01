import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from utils import letter_resize_img
import torchvision


class TestDataset(Dataset):

    def __init__(self, datadir, img_size):
        self.img_pathes = []
        for p in Path(datadir).iterdir():
            if p.is_file() and p.suffix in [".png", '.jpg']:
                self.img_pathes.append(str(p))
        self.img_size = img_size
        self.num_class = 0
        self.class2label = ['lab' for _ in range(self.num_class)]

    def __len__(self):
        return len(self.img_pathes)

    def __iter__(self):
        self.count = 0
        return self

    @staticmethod
    def normalization(img):
        # 输入图像的格式为(h,w,3)
        assert len(img.shape) == 3 and img.shape[-1] == 3
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        return transforms(img)

    def __getitem__(self, item):
        img_bgr = cv2.imread(self.img_pathes[item])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized, letter_info = letter_resize_img(img_rgb, self.img_size, training=False)
        img_normed = self.normalization(img_resized)
        return img_normed, letter_info


def collector(data_in):
    batch_size = len(data_in)
    imgs = [d[0] for d in data_in]
    infoes = [d[1] for d in data_in]
    h, w = imgs[0].shape[1:]
    img_out = torch.ones(batch_size, 3, h, w)
    resize_infoes_out = []
    for i in range(batch_size):
        img_out[i] = imgs[i]
        resize_infoes_out.append(infoes[i])
    return {'img': img_out, 'resize_info': resize_infoes_out}


def testdataloader(datadir, img_size=640, batch_size=1):
    # 因为在inference模式下使用letter_resize_img函数对输入图片进行resize，不会将所有输入的图像都resize到相同的尺寸，而是只要符合输入网络的要求即可
    # assert batch_size == 1, f"use inference mode, so please set batch size to 1"
    dataset = TestDataset(datadir, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collector)
    return dataloader, dataset


def test(datadir, img_size, batch_size):
    dataloader = testdataloader(datadir, img_size, batch_size)
    mean = torch.tensor([0.485, 0.456, 0.406]).float()
    std = torch.tensor([0.229, 0.224, 0.225]).float()
    for x in dataloader:
        for i in range(batch_size):
            img = x['img'][i]
            info = x['info'][i]
            img = img.permute(1, 2, 0)
            img = (img * std + mean) * 255.
            img_mdy = img.numpy().astype('uint8')
            fig, axes = plt.subplots(1, 2, figsize=[16, 16])
            axes[0].imshow(img_mdy)
            axes[0].set_title(f'{img_mdy.shape[:2]}')
            pad_t, pad_b, pad_l, pad_r = info['pad_top'], info['pad_bottom'], info['pad_left'], info['pad_right']
            img_org = img_mdy[pad_t:(info['org_shape'][0]+pad_t), pad_l:(info['org_shape'][1]+pad_l), :]
            # cv2.resize(img_arr, (dst_w, dst_h))
            img_org = cv2.resize(img_org, tuple(info['org_shape'][::-1]), interpolation=0)
            axes[1].imshow(img_org)
            axes[1].set_title(f"{img_org.shape[:2]}")
            plt.show()
            plt.close('all')
            plt.clf()


if __name__ == '__main__':
    test('/Temp/', 640, 1)




