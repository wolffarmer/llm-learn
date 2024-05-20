#!/bin/bash

cur_path=`pwd`

if [ -f "$cur_path/model/tiny-llama.om" ]; then
    echo "[cur_path/model/tiny-llama.om] 已存在 "
else
    cd $cur_path/model
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/tiny_llama/tiny-llama.om
    echo "tiny_llama.om下载完成"
fi

if [ -f "$cur_path/tokenizer/tokenizer.zip" ]; then
    echo "[cur_path/tokenizer/tokenizer.zip] 已存在 "
else
    cd $cur_path/tokenizer
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/wanzutao/tiny-llama/tokenizer.zip
    unzip tokenizer.zip
    echo "tokenizer文件解压完成"
fi

cd $cur_path
pip3 install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple