import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model.model import Model
from dataset.mnist import MnistDataset

def draw_curve(data, name):
    plt.figure()
    plt.plot(data)
    plt.legend([name])
    plt.xlabel('epochs')
    plt.title(f'{name} Curve On Train Dataset')
    plt.savefig(f'./{name}.png')

def train(args):
    # get model
    reg_model = Model(num_classes=args.num_classes, num_instances=args.bag_size).cuda()
    # get train data
    tsf_train = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor(),  # 转换成张量 
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_data = MnistDataset(
        data_root='/mnt/lustre/chenzeren/qzw/project/ntu-biomedical-data-science/project1/data/mnist',
        data_split='train',
        transforms=tsf_train,
        bag_size=args.bag_size)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    mse_best = 1e9
    params = [p for p in reg_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr)
    criterion = torch.nn.L1Loss()
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch)

    all_loss = []
    all_mse = []
    for num_epoch in range(args.epoch):
        epoch_loss = []
        reg_model.train()
        for iter, data in enumerate(train_dataloader):
            imgs, ratios = data

            imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]).cuda()
            ratios = ratios.cuda()
            
            optimizer.zero_grad()
            y_logits = reg_model(imgs).view(-1)

            loss = criterion(y_logits, ratios)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        loss_item = np.array(epoch_loss).mean()
        print(f'Train result : Epoch:{num_epoch} , lr:{scheduler.get_last_lr()[0]:.4f}, loss:{loss_item:.4f}')
        sys.stdout.flush()
        all_loss.append(loss_item)
        
        scheduler.step()
        mse = test(reg_model, args)
        all_mse.append(mse)
        
        if mse < mse_best:
            torch.save(reg_model, './best_model.pth')
            mse_best = mse
        
        print(f"Test result : Epoch:{num_epoch} , mse:{mse:.4f}, best mse:{mse_best:.4f}")
        sys.stdout.flush()

    return all_loss, all_mse

@torch.no_grad()
def test(model, args):
    # get val data
    tsf_test = torchvision.transforms.Compose([ 
        torchvision.transforms.ToTensor(),  # 转换成张量 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    test_data = MnistDataset(
        data_root='/mnt/lustre/chenzeren/qzw/project/ntu-biomedical-data-science/project1/data/mnist',
        data_split='test',
        transforms=tsf_test,
        bag_size=args.bag_size)

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        drop_last=True
    )
    model.eval()
    mse_all = []

    for data in test_dataloader:
        imgs, ratios = data
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]).cuda()
        ratios = ratios.cuda()
        y_logits = model(imgs).view(-1)
        mse = pow((y_logits - ratios), 2).mean()
        mse_all.append(mse.item())

    return np.array(mse_all).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--bag_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=80)
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # train the network
    loss, mse = train(args)

    # draw the loss and mse curve
    draw_curve(loss, 'Loss')
    draw_curve(mse, 'MSE')


