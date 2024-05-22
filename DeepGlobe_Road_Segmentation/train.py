import os
import torch
from torch.nn.parallel import DataParallel
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from arch.dlinknet import DinkNet34
from loss import dice_bce_loss
from data_loader import DeepGlobeRoadExtract
from tqdm import tqdm
from utils import log


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='dlinknet34')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--flag', type=str, default='none')

    return parser.parse_args()


def train_one_epoch(train_loader, model, criterion, optimizer):
    model.train()
    tbar = tqdm(train_loader)
    loss_sum_value = 0.0
    for batch_idx, (img, mask) in enumerate(tbar):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred_mask = model(img)
        loss = criterion(mask, pred_mask)
        loss.backward()
        optimizer.step()
        loss_sum_value += loss.data.cpu()

        tbar.set_description(
            f"epoch: {epoch} / {total_epochs} step: {batch_idx} / {len(train_loader)} "
            f"loss: {loss_sum_value / (batch_idx + 1):.6f}"
        )
    train_loss = loss_sum_value / (len(train_loader))
    return train_loss


def validate(valid_loader, model, criterion):
    model.eval()
    loss_sum_value = 0.0
    for img, mask in tqdm(valid_loader):
        img, mask = img.to(device), mask.to(device)
        pred_mask = model(img)
        loss = criterion(mask, pred_mask)
        loss_sum_value += loss.data.cpu()

    valid_loss = loss_sum_value / (len(valid_loader))
    return valid_loss


if __name__ == '__main__':
    args = get_args()
    device_ids = list(range(torch.cuda.device_count()))

    if args.arch == 'dlinknet34':
        model = DinkNet34(num_classes=1)
    else:
        raise ValueError("Wrong architecture.")
    
    device = torch.device('cuda:0')
    model = DataParallel(model, device_ids=device_ids[:2]).to(device)
    
    ckpt_path = f'./checkpoint/{args.arch}_{args.flag}'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    log_name = f'{args.arch}_{args.flag}.log'

    # record necessary information
    log(f'arch          : {args.arch}\n'
        f'batch-size    : {args.bs}\n'
        f'learning-rate : {args.lr}\n',
        log_path=ckpt_path, log_name=log_name, notime=True)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999))
    # criterion
    criterion = dice_bce_loss()
    # lr scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    # dataset
    training_set = DeepGlobeRoadExtract(mode='train')
    valid_set = DeepGlobeRoadExtract(mode='valid')
    train_loader = data.DataLoader(training_set, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # training procedure
    total_epochs = args.epochs
    best_loss = 100.0
    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer)
        valid_loss = validate(valid_loader, model, criterion)
        for g in optimizer.param_groups:
            curr_lr_value = g['lr']

        log(f"epoch: {epoch} - train loss: {train_loss:.6f} - valid loss: {valid_loss:.6f} - lr: {curr_lr_value:.6f}",
            log_path=ckpt_path, log_name=log_name, print_info=True)
        
        lr_scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'{args.arch}_best.pth'))

    
