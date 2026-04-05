#!/bin/bash
# 00_upload_data_to_hdfs.sh
# 1. 将原始 npz 上传到 HDFS（一次即可）
# 2. 执行 Spark 预处理作业

# 上传 npz（如果尚未上传）
hdfs dfs -test -e /user/wushihang/dgraph/raw/dgraphfin.npz
if [ $? -ne 0 ]; then
    echo "正在上传 npz 文件到 HDFS..."
    bash bin/00_upload_npz_to_hdfs.sh
fi

# 运行预处理作业
echo "启动 Spark 预处理..."
cd /home/wushihang/dgraph_fraud_detection
spark-submit \
  --master local[4] \
  --driver-memory 4g \
  src/01_data_preprocess.py