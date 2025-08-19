#!/bin/bash
# test.sh - 预测入口脚本（科大讯飞复赛提交）

# 清空并创建预测结果目录
rm -rf ../prediction_result/*
mkdir -p ../prediction_result

# 执行Python预测脚本
python predict.py 
