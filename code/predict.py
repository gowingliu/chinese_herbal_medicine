import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import time
import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
import timm
from tqdm import tqdm
from torchvision.transforms import FiveCrop

label_index = pd.read_csv("../xfdata/chinese_herbal_medicine.csv")
label_index = {k:v for v,k in label_index["category"].to_dict().items()}
test_path = glob.glob('../xfdata/test_set_B/*.jpg')
test_path.sort()
def predict(test_path, model):
    model.eval()

    # --- Ten-Crop TTA 实现 ---
    # 首先将图像缩放到比裁剪尺寸稍大
    # 注意：FiveCrop 会返回一个包含5个张量的元组 (tuple)
    tta_transform = transforms.Compose([
        transforms.Resize(420), # 缩放到比 384 稍大的尺寸
        FiveCrop(384), # 四个角 + 中心裁剪
        # Lambda 作用于 5 个 crop 出来的图像
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t) for t in tensors])),
    ])

    test_dataset = TestDataset(test_path, transform=tta_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=4, # 使用 TenCrop 时 batch_size 需要更小
        shuffle=False, 
        num_workers=32,
        pin_memory=True
    )
    
    all_preds_tta = []
    
    # --- 迭代两次：一次原图，一次翻转图 ---
    for flip in [False, True]:
        current_preds = []
        with torch.no_grad():
            for i, (inputs, _) in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Ten-Crop (Flip={flip})"):
                # inputs 的尺寸是 [bs, 5, 3, 384, 384] (bs=batch_size)
                bs, n_crops, c, h, w = inputs.size()
                
                # 如果需要翻转
                if flip:
                    inputs = torch.flip(inputs, dims=[4]) # 在宽度维度上翻转
                
                # 将 batch 和 crops 维度合并，以便一次性送入模型
                inputs = inputs.view(-1, c, h, w).cuda() # 尺寸变为 [bs * 5, 3, 384, 384]
                
                outputs = model(inputs)
                
                # 将结果重新整形，并对 5 个 crop 的结果取平均
                outputs = outputs.view(bs, n_crops, -1).mean(1) # 尺寸变回 [bs, num_classes]
                outputs = F.softmax(outputs, dim=1)
                
                current_preds.append(outputs.cpu().numpy())
                
        all_preds_tta.append(np.vstack(current_preds))

    # 对原图和翻转图的结果进行平均
    avg_preds = np.mean(all_preds_tta, axis=0)
    return avg_preds
# 创建测试集Dataset
class TestDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        try:
            img = Image.open(self.img_path[index]).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color='white')
        
        # 返回文件名用于结果保存
        file_name = os.path.basename(self.img_path[index])
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, file_name
    
    def __len__(self):
        return len(self.img_path)

# 测试集数据转换

# 创建测试集DataLoader
test_dataset = TestDataset(test_path, transform=None)
final_tta_preds = None
N_SPLITS = 5
for fold_num in range(1, N_SPLITS + 1):
    print(f"Predicting with model from fold {fold_num}...")
    model = timm.create_model('convnext_large_in22ft1k', num_classes=len(label_index))
    #model = timm.create_model('swinv2_base_window12to24_192to384_22kft1k', num_classes=len(label_index))
    #model.load_state_dict(torch.load(f'final_model_swa_convnext_fold_{fold_num}.pth'))
    model.load_state_dict(torch.load(f'../user_data/best_convnext_fold_{fold_num}.pth'))

    model = model.cuda()
    
    # 对当前模型进行 TTA 预测
    preds_tta = predict(test_path, model)
    
    if final_tta_preds is None:
        final_tta_preds = preds_tta
    else:
        final_tta_preds += preds_tta

# 对 K 个模型的结果取平均
final_preds = final_tta_preds / N_SPLITS
final_labels = np.argmax(final_preds, axis=1)

# 提取文件名
file_names = [file_name for _, file_name in test_dataset]

# 确保文件名和预测结果数量一致
assert len(file_names) == len(final_labels)

# 创建结果DataFrame
results = pd.DataFrame({
    'ImageID': file_names,
    'label': final_labels
})

# 保存为CSV文件
results.to_csv('../prediction_result/submission_ema.csv', index=False)
print("预测结果已保存到 prediction_result")

# # 显示前10个结果
# print("前10个预测结果:")
# print(results.head(10))