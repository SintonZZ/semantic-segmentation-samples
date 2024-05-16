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


def train_one_batch(model, optimizer, criterion, img, mask):
    model.train()
    optimizer.zero_grad()
    pred_mask = model(img)
    loss = criterion(mask, pred_mask)
    loss.backward()
    optimizer.step()
    return loss


if __name__ == '__main__':
    args = get_args()
    device_ids = list(range(torch.cuda.device_count()))

    if args.arch == 'dlinknet34':
        model = DinkNet34()
    else:
        raise ValueError("Wrong architecture.")
    
    device = torch.device('cuda:0')
    model = DataParallel(model, device_ids=device_ids[:4]).to(device)
    
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
    lr_scheduler = ReduceLROnPlateau(optimizer, 
                                     mode='min',
                                     factor=0.2,
                                     patience=5,
                                     verbose=True)

    # dataset
    training_set = DeepGlobeRoadExtract(mode='train')
    # valid_set = DeepGlobeRoadExtract(mode='valid')
    train_loader = data.DataLoader(training_set,
                                   batch_size=args.bs,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True)
    total_epochs = args.epochs
    best_loss = 100.0
    for epoch in range(1, total_epochs + 1):
        tbar = tqdm(train_loader)
        curr_lr_value = args.lr
        loss_sum_value = 0.0
        for batch_idx, (img, mask) in enumerate(tbar):
            img, mask = img.to(device), mask.to(device)
            loss_vaule = train_one_batch(model, optimizer, criterion, img, mask)
            loss_sum_value += loss_vaule.data.cpu()

            for g in optimizer.param_groups:
                curr_lr_value = g['lr']
            
            tbar.set_description(
                f"Epoch: {epoch} / {total_epochs} Step: {batch_idx} / {len(train_loader)} "
                f"Loss: {loss_sum_value / (batch_idx + 1):.6f} lr: {curr_lr_value:.6f}"
            )
        
        train_loss = loss_sum_value / (len(train_loader))

        log(f"Epoch: {epoch} Loss: {train_loss:.6f} lr: {curr_lr_value:.6f}",
            log_path=ckpt_path, log_name=log_name, print_info=False)
        
        lr_scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'{args.arch}_best.pth'))

    
