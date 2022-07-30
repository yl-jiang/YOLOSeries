import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from pathlib import Path
import cv2


class AuxiliaryClassifierDataset(Dataset):

    def __init__(self, img_dir, img_size):
        super(AuxiliaryClassifierDataset, self).__init__()
        self.img_dir = img_dir
        self.img_size = img_size  # (h, w)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.db = self._build_db(img_dir)

    @staticmethod
    def _build_db(img_dir):
        db = []
        for path in Path(img_dir).iterdir():
            assert path.is_file()
            assert path.suffix in (".png", '.jpg')
            cls = path.name.split('_')[2][5:]
            lab = path.name.split('_')[3]
            db.append((str(path), cls, lab))
        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, item):
        img_path = self.db[item][0]
        cls = self.db[item][1]
        lab = self.db[item][2]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if list(img.shape[:2]) != list(self.img_size):
            img = cv2.resize(img, dsize=tuple(self.img_size[::-1]))
        return self.trans(img), cls, lab


def collector(data_in):
    batch_size = len(data_in)
    img_h, img_w = data_in[0][0].shape[1:]
    img_out = torch.zeros(batch_size, 3, img_h, img_w)
    cls_out = []
    lab_out = []
    for i in range(batch_size):
        img_out[i] = data_in[i][0]
        cls_out.append(int(data_in[i][1]))
        lab_out.append(data_in[i][2])
    return {'img': img_out, 'cls': cls_out, 'lab': lab_out}


def auxiliary_classifier_dataloader(img_dir, img_size, batch_size):
    dataset = AuxiliaryClassifierDataset(img_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collector)
    return dataloader


def test(img_dir, img_size, batch_size):
    dataloader = auxiliary_classifier_dataloader(img_dir, img_size, batch_size)
    mean = torch.tensor([0.485, 0.456, 0.406]).float()
    std = torch.tensor([0.229, 0.224, 0.225]).float()
    for x in dataloader:
        for i in range(batch_size):
            img = x['img'][i]
            cls = x['cls'][i]
            lab = x['lab'][i]
            img = img.permute(1,2,0)
            img = (img * std + mean) * 255.
            img_mdy = img.numpy().astype('uint8')
            fig = plt.figure(figsize=[10, 10])
            plt.imshow(img_mdy)
            plt.title(f"{cls}_{lab}")
            plt.show()
            plt.close('all')


if __name__ == '__main__':
    img_dir = "/img/"
    img_size = [224, 224]
    batch_size = 8
    test(img_dir, img_size, batch_size)