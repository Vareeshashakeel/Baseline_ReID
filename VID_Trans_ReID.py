import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

from Dataloader import dataloader
from VID_Trans_model import VID_Trans
from Loss_fun import make_loss
from VID_Test import test
from utility import AverageMeter, optimizer, scheduler


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID-Trans-ReID no-camera baseline training')
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help='ViT pretrained weight path')
    parser.add_argument('--output_dir', default='./output_camera_removed_corrected', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1234)

    train_loader, _, num_classes, camera_num, view_num, q_val_loader, g_val_loader = dataloader(
        args.Dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, seq_len=args.seq_len
    )
    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.model_path)

    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer_main = optimizer(model)
    lr_scheduler = scheduler(optimizer_main)
    scaler = amp.GradScaler()

    device = 'cuda'
    model = model.to(device)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    best_rank1 = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        lr_scheduler.step(epoch)
        model.train()

        for iteration, (img, pid, target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer_main.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device).view(-1)
            labels2 = labels2.to(device)

            with amp.autocast(enabled=True):
                score, feat, a_vals = model(img, pid, cam_label=target_cam)
                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()
                loss_id, center = loss_fun(score, feat, pid, target_cam)
                loss = loss_id + 0.0005 * center + attn_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer_main)
            scaler.update()
            ema.update()

            for param in center_criterion.parameters():
                if param.grad is not None:
                    param.grad.data *= (1.0 / 0.0005)
            scaler.step(optimizer_center)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            torch.cuda.synchronize()
            if iteration % 50 == 0:
                print('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                    epoch, iteration, len(train_loader), loss_meter.avg, acc_meter.avg, lr_scheduler._get_lr(epoch)[0]
                ))

        print('Epoch {} finished in {:.1f}s'.format(epoch, time.time() - start_time))

        if epoch % args.eval_every == 0:
            model.eval()
            rank1, mAP = test(model, q_val_loader, g_val_loader)
            print('CMC: %.4f, mAP : %.4f' % (rank1, mAP))
            latest_path = os.path.join(args.output_dir, f'{args.Dataset_name}_camera_removed_latest.pth')
            torch.save(model.state_dict(), latest_path)
            if best_rank1 < rank1:
                best_rank1 = rank1
                best_path = os.path.join(args.output_dir, f'{args.Dataset_name}_camera_removed_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint: {best_path}')
