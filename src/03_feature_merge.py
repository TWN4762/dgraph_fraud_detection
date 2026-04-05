#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array
from src import config_loader

def main():
    cfg = config_loader.load_config()
    
    spark = SparkSession.builder \
        .appName("DGraphFin_FeatureMerge") \
        .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.driver.memory", cfg["spark"]["driver_memory"]) \
        .config("spark.sql.shuffle.partitions", cfg["spark"]["shuffle_partitions"]) \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # 读取原始节点表（特征和标签）
    nodes = spark.table(f"dgraph_db.{cfg['data']['table_nodes']}")
    # 读取图特征
    graph_features = spark.table("dgraph_db.dgraph_graph_features")
    
    # 合并：保留 node_id, features, label, 以及图特征
    merged = nodes.join(graph_features, on="node_id", how="inner")
    
    # 将 features 数组和各个图特征拼接成一个特征向量（用于 ML）
    # 注意：features 已经是 ArrayType(FloatType)，我们保留原始特征列
    # 训练时需要将所有特征列（包括图特征）组装为向量，但这一步只负责合并表
    # 我们保留所有列，在模型训练时选择列即可
    
    # 只保留 label 为 0 或 1 的前景节点（排除背景节点 -1）
    merged = merged.filter(col("label").isin([0, 1]))
    
    print(f"合并后前景节点数: {merged.count()}")
    
    # 写入 Hive 宽表
    merged.write.mode("overwrite").saveAsTable("dgraph_db.dgraph_merged_features")
    print("合并特征表已保存至 dgraph_db.dgraph_merged_features")
    
    spark.stop()

if __name__ == "__main__":
    main()