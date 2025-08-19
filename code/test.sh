#!/bin/bash
# test.sh - 预测入口脚本（科大讯飞复赛提交）

# 检查模型文件是否存在
MODEL_PATH="../user_data/fusion_model_data/best_model_fusion_feature.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# 清空并创建预测结果目录
rm -rf ../prediction_result/*
mkdir -p ../prediction_result

# 执行Python预测脚本
python predict.py \
    --data_dir '../xfdata/xfdata/preliminary_test' \
    --model_path "$MODEL_PATH" \
    --output_dir ../prediction_result \
