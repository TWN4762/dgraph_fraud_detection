#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from src import config_loader

def main():
    cfg = config_loader.load_config()
    
    spark = SparkSession.builder \
        .appName("DGraphFin_Visualize") \
        .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 读取图特征表（包含连通分量）
    graph_features = spark.table("dgraph_db.dgraph_graph_features")
    
    # 1. 连通分量（团伙）规模分布
    cc_counts = graph_features.groupBy("component").count().orderBy("count", ascending=False).toPandas()
    top_cc = cc_counts.head(20)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_cc, x="component", y="count", palette="viridis")
    plt.title("Top 20 Connected Components (Fraud Groups)")
    plt.xlabel("Component ID")
    plt.ylabel("Number of Nodes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/figures/component_distribution.png", dpi=cfg["viz"]["dpi"])
    plt.close()
    print("团伙规模分布图已保存至 output/figures/component_distribution.png")
    
    # 2. PageRank 分布直方图
    pr_df = graph_features.select("pagerank").toPandas()
    plt.figure(figsize=(10, 6))
    sns.histplot(pr_df["pagerank"], bins=50, kde=True, log_scale=True)
    plt.title("PageRank Distribution (Log Scale)")
    plt.xlabel("PageRank")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/figures/pagerank_distribution.png", dpi=cfg["viz"]["dpi"])
    plt.close()
    print("PageRank 分布图已保存至 output/figures/pagerank_distribution.png")
    
    # 3. 节点度分布（入度+出度）
    deg_df = graph_features.select("inDegree", "outDegree", "degree").toPandas()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(deg_df["inDegree"], bins=50, log_scale=True, ax=axes[0])
    axes[0].set_title("In-Degree Distribution")
    sns.histplot(deg_df["outDegree"], bins=50, log_scale=True, ax=axes[1])
    axes[1].set_title("Out-Degree Distribution")
    sns.histplot(deg_df["degree"], bins=50, log_scale=True, ax=axes[2])
    axes[2].set_title("Total Degree Distribution")
    plt.tight_layout()
    plt.savefig("output/figures/degree_distribution.png", dpi=cfg["viz"]["dpi"])
    plt.close()
    print("节点度分布图已保存至 output/figures/degree_distribution.png")
    
    spark.stop()

if __name__ == "__main__":
    main()