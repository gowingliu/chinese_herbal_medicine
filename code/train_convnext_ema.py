# -*- coding: utf-8 -*-
import os, glob, time, math, random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm

import timm
from timm.data import Mixup
from torchvision import transforms
from torchvision.transforms import RandAugment

# =======================
# 1. 全局配置 
# =======================
SEED = 42
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
TRAIN_DIR = '../xfdata/train_set/'
CSV_PATH  = '../xfdata/chinese_herbal_medicine.csv'

# 修改为ConvNeXt模型
MODEL_NAME = 'convnext_large_in22ft1k'
IMG_SIZE = 384
N_SPLITS = 5

BATCH_SIZE = 8
NUM_WORKERS = 8

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 25
WARMUP_EPOCHS = 3

BASE_LR   = 5e-5
HEAD_LR_P1 = 1e-4
WEIGHT_DECAY = 1e-3

# LLRD: ConvNeXt各层学习率倍率
# ConvNeXt结构: stages.0, stages.1, stages.2, stages.3, head
LLRD_STAGE_LRS = {
    'stages.0': 0.1,
    'stages.1': 0.25,
    'stages.2': 0.5,
    'stages.3': 0.75,
    'head'    : 1.0
}

MIXUP_ALPHA = 0.4; CUTMIX_ALPHA = 0.6; MIXUP_PROB = 0.5;
RE_PROB = 0.25 # RandomErasing 概率
USE_EMA = True; EMA_DECAY = 0.999

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = False;
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

# =======================
# 2. 数据处理
# =======================
def read_label_map(csv_path):
    df = pd.read_csv(csv_path); return {cat: i for i, cat in enumerate(df['category'])}

def load_paths_and_labels(train_dir, label_to_index):
    paths = glob.glob(os.path.join(train_dir, '*/*.jpg')); random.shuffle(paths)
    valid_paths, labels = [], []
    for p in paths:
        try:
            Image.open(p).convert('RGB').size; valid_paths.append(p)
            labels.append(label_to_index[os.path.basename(os.path.dirname(p))])
        except: 
            print(f"跳过损坏的图像: {p}")
            pass
    return np.array(valid_paths), np.array(labels)

class XFDataset(Dataset):
    def __init__(self, img_paths, labels, transform): self.img_paths,self.labels,self.transform = img_paths,labels,transform
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        try: img = Image.open(self.img_paths[idx]).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        return self.transform(img), int(self.labels[idx])

def build_transforms(img_size=IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        RandAugment(num_ops=2, magnitude=10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=RE_PROB, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

# =================================================================
# 3. 工具：优化器分组(LLRD)、EMA、调度器 
# =================================================================
def no_weight_decay_param(name):
    return any(k in name.lower() for k in ['bias','gamma','beta','norm','bn','ln'])

def build_llrd_param_groups(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY):
    """为ConvNeXt构建分层学习率的参数组"""
    param_groups = []; named_params = list(model.named_parameters())
    for name, p in named_params:
        if not p.requires_grad: continue
        lr_mult = 1.0
        
        # ConvNeXt层名匹配规则
        if 'head' in name: 
            lr_mult = LLRD_STAGE_LRS['head']
        elif 'stages.3' in name: 
            lr_mult = LLRD_STAGE_LRS['stages.3']
        elif 'stages.2' in name: 
            lr_mult = LLRD_STAGE_LRS['stages.2']
        elif 'stages.1' in name: 
            lr_mult = LLRD_STAGE_LRS['stages.1']
        elif 'stages.0' in name: 
            lr_mult = LLRD_STAGE_LRS['stages.0']
        
        lr = base_lr * lr_mult
        wd = 0.0 if no_weight_decay_param(name) else weight_decay
        param_groups.append({'params': [p], 'lr': lr, 'weight_decay': wd})
        
    return param_groups

class WarmupCosine:
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, min_lr=1e-7):
        self.optimizer, self.total_epochs, self.warmup_epochs, self.min_lr = optimizer, total_epochs, warmup_epochs, min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
    def step(self, epoch):
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if epoch < self.warmup_epochs: lr = base_lr * float(epoch + 1) / float(self.warmup_epochs)
            else: t = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs); lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
            group['lr'] = lr

class ModelEMA:
    """指数移动平均"""
    def __init__(self, model, decay=EMA_DECAY):
        import copy
        self.ema = copy.deepcopy(model); self.ema.eval(); self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point: v.mul_(self.decay).add_(msd[k].detach(), alpha=1. - self.decay)
    def state_dict(self): return self.ema.state_dict()

# =======================
# 4. 训练/验证 (适配EMA)
# =======================
def train_one_epoch(train_loader, model, criterion, optimizer, scaler, mixup_fn=None, ema_obj=None):
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.cuda(non_blocking=True), targets.long().cuda(non_blocking=True)
        if mixup_fn: images, targets = mixup_fn(images, targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images); loss = criterion(outputs, targets)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if ema_obj: ema_obj.update(model)

@torch.no_grad()
def validate(val_loader, model, criterion):
    model.eval(); all_tgts, all_preds = [], []; total_loss = 0.0
    for images, targets in tqdm(val_loader, total=len(val_loader), desc='Validating'):
        images, targets = images.cuda(), targets.long().cuda()
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        all_tgts.extend(targets.cpu().numpy()); all_preds.extend(outputs.argmax(1).cpu().numpy())
    macro_f1 = f1_score(all_tgts, all_preds, average='macro'); avg_loss = total_loss / len(val_loader.dataset)
    print(f' * Val Loss: {avg_loss:.4f} | Macro-F1: {macro_f1:.4f}')
    return macro_f1, avg_loss

# =======================
# 5. 主流程 
# =======================
def main():
    set_seed(SEED)
    label_to_index = read_label_map(CSV_PATH)
    num_classes = len(label_to_index)
    all_paths, all_labels = load_paths_and_labels(TRAIN_DIR, label_to_index)
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_f1_scores = []
    train_tf, val_tf = build_transforms(IMG_SIZE)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_paths), 1):
        print(f"\n========== FOLD {fold}/{N_SPLITS} ==========")
        train_paths, val_paths = all_paths[tr_idx], all_paths[va_idx]
        train_labels, val_labels = all_labels[tr_idx], all_labels[va_idx]
        train_loader = torch.utils.data.DataLoader(
            XFDataset(train_paths, train_labels, train_tf), 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS, 
            pin_memory=True, 
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            XFDataset(val_paths, val_labels, val_tf), 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )
        
        # 创建ConvNeXt模型
        model = timm.create_model(
            MODEL_NAME, 
            pretrained=True,  
            num_classes=num_classes, 
            drop_path_rate=0.3
        ).cuda()
        print(f"创建ConvNeXt模型: {MODEL_NAME}, 分类数: {num_classes}")
        
        # --- 创建EMA对象 ---
        ema_obj = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None
        
        # --- Phase 1：仅训分类头 ---
        print("\n--- Phase 1: Train classifier head ---")
        # 冻结除分类头外的所有层
        for name, p in model.named_parameters():
            p.requires_grad = 'head' in name
        
        optimizer_head = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=HEAD_LR_P1, 
            weight_decay=WEIGHT_DECAY
        )
        scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_head, 
            T_max=PHASE1_EPOCHS, 
            eta_min=1e-6
        )
        mixup_fn = Mixup(
            mixup_alpha=MIXUP_ALPHA, 
            cutmix_alpha=CUTMIX_ALPHA, 
            prob=MIXUP_PROB, 
            switch_prob=0.5, 
            mode='batch', 
            label_smoothing=0.05, 
            num_classes=num_classes
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        best_f1_fold = 0.0

        for epoch in range(PHASE1_EPOCHS):
            print(f"Epoch {epoch+1}/{PHASE1_EPOCHS} (Head)")
            train_one_epoch(train_loader, model, criterion, optimizer_head, scaler, mixup_fn=mixup_fn, ema_obj=ema_obj)
            # --- 验证时使用EMA模型 ---
            eval_model = ema_obj.ema if USE_EMA else model
            f1, _ = validate(val_loader, eval_model, criterion)
            scheduler_head.step()
            if f1 > best_f1_fold:
                best_f1_fold = f1
                # --- 保存EMA模型权重 ---
                torch.save(
                    (ema_obj.state_dict() if USE_EMA else model.state_dict()), 
                    f'../user_data/best_convnext_fold_{fold}.pth'
                )
                print(f"  * New best (Phase1) F1: {best_f1_fold:.4f} -> saved")

        # --- Phase 2：全参微调 (使用LLRD) ---
        print("\n--- Phase 2: Full fine-tuning with LLRD ---")
        # 解冻所有层
        for p in model.parameters(): 
            p.requires_grad = True
        
        # --- 加载最佳EMA权重并重新初始化EMA对象 ---
        if USE_EMA:
            print(f"加载阶段1的最佳EMA权重: best_convnext_fold_{fold}.pth")
            model.load_state_dict(torch.load(f'best_convnext_fold_{fold}.pth'), strict=True)
            ema_obj = ModelEMA(model, decay=EMA_DECAY)

        # --- 使用LLRD构建优化器 ---
        param_groups = build_llrd_param_groups(model, base_lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        optimizer_full = optim.AdamW(param_groups)
        scheduler_full = WarmupCosine(
            optimizer_full, 
            total_epochs=PHASE2_EPOCHS, 
            warmup_epochs=WARMUP_EPOCHS, 
            min_lr=1e-7
        )

        for epoch in range(PHASE2_EPOCHS):
            print(f"Epoch {epoch+1}/{PHASE2_EPOCHS} (Full)")
            train_one_epoch(train_loader, model, criterion, optimizer_full, scaler, mixup_fn=mixup_fn, ema_obj=ema_obj)
            eval_model = ema_obj.ema if USE_EMA else model
            f1, _ = validate(val_loader, eval_model, criterion)
            scheduler_full.step(epoch)
            if f1 > best_f1_fold:
                best_f1_fold = f1
                torch.save(
                    (ema_obj.state_dict() if USE_EMA else model.state_dict()), 
                    f'../user_data/best_convnext_fold_{fold}.pth'
                )
                print(f"  * New best (Phase2) F1: {best_f1_fold:.4f} -> saved")

        fold_f1_scores.append(best_f1_fold)
        print(f"\nBest F1 for Fold {fold}: {best_f1_fold:.4f}")

    print(f"\nAverage F1 across {N_SPLITS} folds: {np.mean(fold_f1_scores):.4f}")

if __name__ == "__main__":
    main()