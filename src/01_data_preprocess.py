#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tempfile
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, ArrayType, LongType
from pyspark.sql.functions import col, rand

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config_loader

def read_npz_from_hdfs(hdfs_path):
    """
    使用 hadoop 命令将 HDFS 上的 npz 文件拉到本地临时文件，
    然后用 numpy 加载，最后删除临时文件。
    """
    import tempfile
    import subprocess
    import os
    
    # 创建临时文件（mkstemp 返回文件描述符和路径）
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)  # 关闭文件描述符，避免占用
    
    # 使用 -f 选项强制覆盖（如果目标文件已存在）
    cmd = f"hdfs dfs -get -f {hdfs_path} {tmp_path}"
    subprocess.check_call(cmd, shell=True)
    
    # 加载 npz
    data = np.load(tmp_path)
    
    # 删除临时文件
    os.unlink(tmp_path)
    return data

def main():
    cfg = config_loader.load_config()
    
    # 创建 SparkSession（启用 Hive）
    spark = SparkSession.builder \
        .appName("DGraphFin_Import") \
        .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.driver.memory", cfg["spark"]["driver_memory"]) \
        .config("spark.sql.shuffle.partitions", cfg["spark"]["shuffle_partitions"]) \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # 1. 从 HDFS 读取 npz 文件（Driver 端）
    print("正在从 HDFS 加载 npz 文件...")
    npz_path = cfg["data"]["hdfs_npz_path"]
    data = read_npz_from_hdfs(npz_path)
    
    x = data['x']                   # (N, 17)
    y = data['y']                   # (N,)
    edge_index = data['edge_index'] # (E, 2)
    edge_type = data['edge_type']   # (E,)
    edge_timestamp = data['edge_timestamp']  # (E,)
    train_mask = data['train_mask']   # (N_train,)
    valid_mask = data['valid_mask']
    test_mask = data['test_mask']
    
    N = x.shape[0]
    E = edge_index.shape[0]
    print(f"原始节点数: {N}, 原始边数: {E}")
    
    # 2. 采样
    if cfg["data"]["enable_sampling"]:
        frac = cfg["data"]["sample_fraction"]
        seed = cfg["data"]["sample_seed"]
        print(f"采样比例: {frac}, 随机种子: {seed}")
        np.random.seed(seed)
        
        # 获取前景节点索引（训练+验证+测试）
        foreground = np.concatenate([train_mask, valid_mask, test_mask])
        bg_mask = np.ones(N, dtype=bool)
        bg_mask[foreground] = False
        background = np.where(bg_mask)[0]
        
        # 采样前景节点
        n_fore = len(foreground)
        n_sample_fore = int(n_fore * frac)
        sampled_fore = np.random.choice(foreground, n_sample_fore, replace=False)
        
        # 采样背景节点
        n_bg = len(background)
        n_sample_bg = int(n_bg * frac)
        sampled_bg = np.random.choice(background, n_sample_bg, replace=False)
        
        sampled_nodes = np.concatenate([sampled_fore, sampled_bg])
        sampled_nodes.sort()
        
        # 节点重映射
        node_map = {old: new for new, old in enumerate(sampled_nodes)}
        
        # 提取采样节点特征和标签
        x_sampled = x[sampled_nodes]
        y_sampled = y[sampled_nodes]
        
        # 过滤边
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        mask = np.isin(src, sampled_nodes) & np.isin(dst, sampled_nodes)
        src_sampled = src[mask]
        dst_sampled = dst[mask]
        edge_type_sampled = edge_type[mask]
        edge_ts_sampled = edge_timestamp[mask]
        
        # 重映射节点索引
        src_mapped = np.array([node_map[node] for node in src_sampled])
        dst_mapped = np.array([node_map[node] for node in dst_sampled])
        
        print(f"采样后节点数: {len(sampled_nodes)}, 边数: {len(src_mapped)}")
        
        # 更新数据
        x = x_sampled
        y = y_sampled
        src = src_mapped
        dst = dst_mapped
        edge_type = edge_type_sampled
        edge_timestamp = edge_ts_sampled
    
    # 3. 转换为 Spark DataFrame 并写入 Parquet
    # 定义节点表的 schema
    node_schema = StructType([
        StructField("node_id", IntegerType(), True),
        StructField("features", ArrayType(FloatType()), True),
        StructField("label", IntegerType(), True)
    ])

    # 构建节点数据，并将 numpy float64 转换为 Python float
    print("转换节点数据...")
    node_data = []
    for i in range(len(x)):
        node_data.append({
            "node_id": i,
            "features": [float(v) for v in x[i]],   # 关键：转换为 Python float
            "label": int(y[i])
        })
    node_df = spark.createDataFrame(node_data, schema=node_schema)

    # 边数据的schema
    edge_schema = StructType([
        StructField("src", IntegerType(), True),
        StructField("dst", IntegerType(), True),
        StructField("edge_type", IntegerType(), True),
        StructField("edge_timestamp", IntegerType(), True)
    ])

    print("转换边数据...")
    edge_data = [
        {"src": int(src[i]), "dst": int(dst[i]), 
        "edge_type": int(edge_type[i]), 
        "edge_timestamp": int(edge_timestamp[i])}
        for i in range(len(src))
    ]
    edge_df = spark.createDataFrame(edge_data, schema=edge_schema)
    
    # 写入 Parquet 到 HDFS
    hdfs_parquet_path = cfg["data"]["hdfs_parquet_path"]
    print(f"写入节点表到 {hdfs_parquet_path}/nodes")
    node_df.write.mode("overwrite").parquet(f"{hdfs_parquet_path}/nodes")
    print(f"写入边表到 {hdfs_parquet_path}/edges")
    edge_df.write.mode("overwrite").parquet(f"{hdfs_parquet_path}/edges")
    
    # 4. 创建 Hive 外部表
    spark.sql("CREATE DATABASE IF NOT EXISTS dgraph_db")
    spark.sql(f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS dgraph_db.{cfg['data']['table_nodes']}
        (node_id INT, features ARRAY<FLOAT>, label INT)
        STORED AS PARQUET
        LOCATION '{hdfs_parquet_path}/nodes'
    """)
    spark.sql(f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS dgraph_db.{cfg['data']['table_edges']}
        (src INT, dst INT, edge_type INT, edge_timestamp INT)
        STORED AS PARQUET
        LOCATION '{hdfs_parquet_path}/edges'
    """)
    
    print("数据导入完成！")
    spark.stop()

if __name__ == "__main__":
    main()