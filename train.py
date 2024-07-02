import os
import random
import monai
from os import makedirs
from os.path import join
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn_extra.cluster import KMedoids 
from typing import Optional, Tuple

# import sys
# sys.path.append('/home/zijianwu/projects/def-timsbc/zijianwu/codes/MedSAM/')
from mobile_sam.build_sam import sam_model_registry
from surgical_tool_sam import SurgicalToolSAM
from dataset import FinetuneDataset

import cv2
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr-npy-path', type=str, help="Path to the training data root directory.", required=True)
parser.add_argument('-v', '--val-npy-path', type=str, help="Path to the validation data root directory.", required=True)
parser.add_argument('--sam-ckpt', type=str, help="Path to the SAM checkpoint.", required=True)
parser.add_argument('--work-dir', type=str, default="finetune_point_prompt", help="Path to where the checkpoints and logs are saved.")
parser.add_argument('--max-epochs', type=int, default=1000, help="Maximum number of epochs.")
parser.add_argument('-bs','--batch-size', type=int, default=16, help="Batch size.")
parser.add_argument('--num-workers', type=int, default=8, help="Number of data loader workers.")
parser.add_argument('--learn-rate', type=float, default=0.00005, help="learning rate (absolute lr)")
parser.add_argument('-wd', '--weight-decay', type=float, default=0.01, help="Weight decay.")
parser.add_argument('--seed', type=int, default=2023, help="Random seed for reproducibility.")
parser.add_argument('--data-aug', action="store_true", help="Enable data augmentation.")
parser.add_argument('--freeze-image-encoder', action="store_true", help="freeze image encoder or not")
parser.add_argument('--freeze-prompt-encoder', action="store_true", help="freeze prompt encoder or not")
parser.add_argument('--freeze-mask-decoder', action="store_true", help="freeze mask decoder or not")
parser.add_argument('--multi-dataset', action="store_true", help='if use multiple datasets')
parser.add_argument('--train-from-scratch', action="store_true", help='Train from scratch or not')
parser.add_argument('--dataset', type=str, help='The name abbreviation of the dataset')
parser.add_argument('--multi-gpu', action='store_true', help='The number of multiple GPU for training')

args = parser.parse_args()

data_root = args.tr_npy_path
val_data_root = args.val_npy_path
work_dir = args.work_dir
num_epochs = args.max_epochs
batch_size = args.batch_size
num_workers = args.num_workers
sam_ckpt = args.sam_ckpt
data_aug = args.data_aug
seed = args.seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda:0"
makedirs(work_dir, exist_ok=True)

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

mobile_sam_ckpt = './ckpts/mobile_sam.pt'         
surgicaltool_sam = SurgicalToolSAM(
    ckpt=mobile_sam_ckpt,#sam_ckpt, 
    freeze_image_encoder=False, 
    freeze_prompt_encoder=True,
    freeze_mask_decoder=False,
)
if not args.train_from_scratch:
    sam_ckpt = torch.load(sam_ckpt)
    surgicaltool_sam.load_state_dict(sam_ckpt['model'])

if args.multi_gpu:
    surgicaltool_sam = nn.DataParallel(surgicaltool_sam, device_ids=[0,1,2,3])
else:
    surgicaltool_sam = surgicaltool_sam.to(device)

surgicaltool_sam.train()
print(f"SAM model size: {sum(p.numel() for p in surgicaltool_sam.parameters())}")

if args.multi_gpu:
    model_params = surgicaltool_sam.module.parameters()
else:
    model_params = surgicaltool_sam.parameters()

optimizer = optim.AdamW(
    model_params,
    lr=args.learn_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=args.weight_decay #0.01
)

start_epoch = 0
best_loss = 1e10

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

train_dataset = FinetuneDataset(data_root=data_root, dataset_name=args.dataset, data_aug=data_aug, status='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_dataset = FinetuneDataset(data_root=val_data_root, dataset_name=args.dataset, data_aug=False, status='val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
torch.cuda.empty_cache()

epoch_time = []
losses = []
val_losses = []
lr_list = []

for epoch in range(start_epoch, num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    surgicaltool_sam.train()
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2D = batch["gt2D"]
        coords_torch = batch["coords"] # (B, N, 2)
        optimizer.zero_grad()
        labels_torch = torch.ones(coords_torch.shape[0], coords_torch.shape[1]).long() # (B, N)
        image, gt2D = image.to(device), gt2D.to(device)
        coords_torch, labels_torch = coords_torch.to(device), labels_torch.to(device)
        point_prompt = (coords_torch, labels_torch)
        medsam_lite_pred = surgicaltool_sam(image, point_prompt, 'training')
        loss = seg_loss(medsam_lite_pred, gt2D) + ce_loss(medsam_lite_pred, gt2D.float())
        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, training loss: {loss.item():.4f}")

    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    losses.append(epoch_loss_reduced)

    if args.multi_gpu:
        model_weights = surgicaltool_sam.module.state_dict()
    else:
        model_weights = surgicaltool_sam.state_dict()

    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss
    }

    torch.save(checkpoint, join(work_dir, "surgicaltoolsam_latest.pth"))

    # validation
    val_epoch_loss = [1e10 for _ in range(len(val_loader))]
    val_pbar = tqdm(val_loader)
    surgicaltool_sam.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            coords_torch = batch["coords"] # (B, N, 2)
            labels_torch = torch.ones(coords_torch.shape[0], coords_torch.shape[1]).long() # (B, N)
            image, gt2D = image.to(device), gt2D.to(device)
            coords_torch, labels_torch = coords_torch.to(device), labels_torch.to(device)
            point_prompt = (coords_torch, labels_torch)
            medsam_lite_pred = surgicaltool_sam(image, point_prompt, 'training')
            loss = seg_loss(medsam_lite_pred, gt2D) + ce_loss(medsam_lite_pred, gt2D.float())
            val_epoch_loss[step] = loss.item()
            val_pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, validation loss: {loss.item():.4f}")

    val_epoch_loss_reduced = sum(val_epoch_loss) / len(val_epoch_loss)
    val_losses.append(val_epoch_loss_reduced)

    if val_epoch_loss_reduced < best_loss:
        print(f"New best validation loss: {best_loss:.4f} -> {val_epoch_loss_reduced:.4f}")
        best_loss = val_epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "surgicaltoolsam_best.pth"))


    epoch_end_time = time()
    epoch_time.append(epoch_end_time - epoch_start_time)

    scheduler.step()
    lr_list.append(scheduler.get_lr()[0])

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 10))
    
    ax1.plot(losses)
    ax1.set_title("TRaining: Dice + Cross Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    
    ax2.plot(val_losses)
    ax2.set_title("Validation: Dice + Cross Entropy Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")

    ax3.plot(epoch_time)
    ax3.set_title("Epoch Running Time")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time (s)")

    ax4.plot(lr_list)
    ax4.set_title("Learning Rate Decay")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    
    fig.savefig(join(work_dir, "medsam_point_prompt_loss_time.png"))

    epoch_loss_reduced = 1e10
    val_epoch_loss_reduced = 1e10