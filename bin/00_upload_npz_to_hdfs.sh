#!/bin/bash
# 将本地 npz 文件上传至 HDFS

LOCAL_NPZ="/home/wushihang/data/dgraph/dgraphfin.npz"
HDFS_NPZ_DIR="/user/wushihang/dgraph/raw"

# 确保 HDFS 目录存在
hdfs dfs -mkdir -p $HDFS_NPZ_DIR
# 上传文件（如果已存在则覆盖）
hdfs dfs -put -f $LOCAL_NPZ $HDFS_NPZ_DIR/

echo "NPZ 文件已上传至 HDFS: $HDFS_NPZ_DIR/dgraphfin.npz"