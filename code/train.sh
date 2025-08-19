#!/bin/bash

python train_convnext_ema.py

echo "========================================"
echo "训练完成！"
echo "模型已保存至: ../user_data/tmp_data/"
echo "完成时间: $(date)"
echo "========================================"