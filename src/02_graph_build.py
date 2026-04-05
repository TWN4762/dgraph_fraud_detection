#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from graphframes import GraphFrame
from src import config_loader

def main():
    cfg = config_loader.load_config()
    
    spark = SparkSession.builder \
        .appName("DGraphFin_GraphBuild") \
        .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.driver.memory", cfg["spark"]["driver_memory"]) \
        .config("spark.sql.shuffle.partitions", cfg["spark"]["shuffle_partitions"]) \
        .enableHiveSupport() \
        .getOrCreate()
    
    # 设置日志级别为 ERROR，减少冗余信息
    spark.sparkContext.setLogLevel("ERROR")
    
    # 设置 checkpoint 目录（连通分量必需）
    checkpoint_dir = "hdfs://localhost:9000/user/wushihang/checkpoint"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    # 确保目录存在
    from subprocess import call
    call(f"hdfs dfs -mkdir -p {checkpoint_dir}", shell=True)
    
    # 读取节点和边表
    nodes_df = spark.table(f"dgraph_db.{cfg['data']['table_nodes']}")
    edges_df = spark.table(f"dgraph_db.{cfg['data']['table_edges']}")
    
    print(f"节点数: {nodes_df.count()}, 边数: {edges_df.count()}")
    
    # 构建 GraphFrame（顶点列名必须为 "id"，边列名为 "src" 和 "dst"）
    vertices = nodes_df.select(col("node_id").alias("id"))
    edges = edges_df.select(col("src").alias("src"), col("dst").alias("dst"))
    
    g = GraphFrame(vertices, edges)
    
    # 1. 连通分量
    print("计算连通分量...")
    cc_result = g.connectedComponents()
    # cc_result 包含列: id, component
    cc_result = cc_result.select(col("id").alias("node_id"), col("component"))
    
    # 2. PageRank
    print("计算 PageRank...")
    pr_result = g.pageRank(resetProbability=0.15, maxIter=10)
    pr_df = pr_result.vertices.select(col("id").alias("node_id"), col("pagerank"))
    
    # 3. 节点度
    print("计算节点度...")
    in_deg = g.inDegrees.select(col("id").alias("node_id"), col("inDegree"))
    out_deg = g.outDegrees.select(col("id").alias("node_id"), col("outDegree"))
    deg = g.degrees.select(col("id").alias("node_id"), col("degree"))
    
    # 合并图特征
    graph_features = nodes_df.select("node_id").distinct()
    graph_features = graph_features.join(cc_result, on="node_id", how="left")
    graph_features = graph_features.join(pr_df, on="node_id", how="left")
    graph_features = graph_features.join(in_deg, on="node_id", how="left")
    graph_features = graph_features.join(out_deg, on="node_id", how="left")
    graph_features = graph_features.join(deg, on="node_id", how="left")
    
    # 填充缺失值为0
    graph_features = graph_features.fillna(0)
    
    # 写入 Hive 表
    graph_features.write.mode("overwrite").saveAsTable("dgraph_db.dgraph_graph_features")
    print("图特征已保存至 dgraph_db.dgraph_graph_features")
    
    spark.stop()

if __name__ == "__main__":
    main()