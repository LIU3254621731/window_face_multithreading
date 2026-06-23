#!/bin/bash

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
cmake ..

# 编译
make -j4

# 返回上级目录
cd ..

echo "Build completed" 