import numpy as np
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataset import auxiliary_classifier_dataloader
from tqdm import tqdm


class AuxiliaryClassifier(torch.nn.Module):

    def __init__(self, classifier_name='resnet101', class_num=80):
        super(AuxiliaryClassifier, self).__init__()
        self.model = models.__dict__[classifier_name](pretrained=True)
        filters = self.model.fc.weight.shape[1]  # fc layer ouoput channel
        self.model.fc = torch.nn.Linear(filters, class_num)
        # self.model.fc.bias = torch.nn.Parameter(torch.zeros(class_num), requires_grad=True)
        # self.model.fc.weight = torch.nn.Parameter(torch.zeros(class_num, filters), requires_grad=True)
        self.model.fc.out_features = class_num

    def forward(self, x):
        preds = self.model(x)
        return preds


def train_aux_classifier(aux_c, train_dir, test_dir):
    train_dataloader = auxiliary_classifier_dataloader(train_dir, [224, 224], 32)
    test_dataloader = auxiliary_classifier_dataloader(test_dir, [224, 224], 32)
    pretrain_param_group, finetune_param_group = [], []
    for k, v in aux_c.named_parameters():
        v.required_grad = True
        if '.fc' in k:
            finetune_param_group.append(v)
        else:
            pretrain_param_group.append(v)
    params = {{'params': pretrain_param_group, 'lr': 1e-6}, {'params': finetune_param_group, 'lr': 1e-3}}
    optimizer = optim.AdamW(params=params, lr=1e-3, betas = (0.937, 0.999))
    loss_fnc = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aux_c = aux_c.to(device)

    total_epoch = 1000
    for epoch in range(total_epoch):
        with tqdm(ncols=120, total=len(train_dataloader)) as tbar:
            tbar.set_description_str(f"Epoch[{epoch}/{total_epoch}]")
            for i, x in enumerate(train_dataloader):
                aux_c.train()
                img = x['img'].to(device)
                lab = torch.tensor(x['cls']).long().to(device)
                preds = aux_c(img)
                loss = loss_fnc(preds, lab)
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()

                cur_step = epoch * len(train_dataloader) + i + 1
                if cur_step % 500 == 0:
                    aux_c.eval()
                    correct_num = 0
                    total_num = 0
                    for j, xtest in enumerate(test_dataloader):
                        img = xtest['img'].to(device)
                        lab = torch.tensor(xtest['cls']).long().to(device)
                        preds = aux_c(img)  # (bs, 80)
                        preds_y = preds.max(dim=1)[1]
                        correct_num += (preds_y == lab).sum()
                        total_num += len(img)
                    acc = correct_num / total_num
                    tbar.set_postfix_str(f"loss = {loss.detach().item():.5f} accuracy = {acc:.5f}")
                tbar.update(1)


if __name__ == '__main__':
    aux_c = AuxiliaryClassifier()
    train_aux_classifier(aux_c, "/classifier_train/", "/classifier_test/")