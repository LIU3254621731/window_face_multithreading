#!/bin/bash

# 设置环境变量
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/firefly/opencv411/lib:$LD_LIBRARY_PATH

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <fps>"
    echo "Example: $0 ../model/RetinaFace.rknn 30"
    exit 1
fi

# 运行程序
./Face_Mesh_v2 $1 $2

# 检查程序执行状态
if [ $? -eq 0 ]; then
    echo "Program executed successfully!"
else
    echo "Program execution failed!"
    exit 1
fi