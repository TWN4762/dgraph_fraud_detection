#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from src import config_loader

def main():
    cfg = config_loader.load_config()
    
    spark = SparkSession.builder \
        .appName("DGraphFin_ModelTrain") \
        .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.driver.memory", cfg["spark"]["driver_memory"]) \
        .config("spark.sql.shuffle.partitions", cfg["spark"]["shuffle_partitions"]) \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # 读取合并后的宽表
    df = spark.table("dgraph_db.dgraph_merged_features")
    
    # 定义特征列：原始 features 数组 + 图特征列
    # features 是 ArrayType，不能直接作为 ML 特征，需要转换为向量
    # 方法：将数组展开为多个列（Spark 2.3+ 不支持直接对数组做 VectorAssembler）
    # 替代方案：使用 UDF 将数组转换为向量，或者将数组元素拆分为独立列
    # 由于特征维度只有17，我们可以将数组拆分为17个独立列，然后与图特征组装
    
    # 获取数组元素个数（应该为17）
    first_row = df.first()
    num_features = len(first_row["features"])
    print(f"原始特征维度: {num_features}")
    
    # 动态生成列名：feat_0 到 feat_{num_features-1}
    from pyspark.sql.functions import col
    for i in range(num_features):
        df = df.withColumn(f"feat_{i}", col("features")[i])
    
    # 图特征列名
    graph_cols = ["component", "pagerank", "inDegree", "outDegree", "degree"]
    # 所有特征列
    feature_cols = [f"feat_{i}" for i in range(num_features)] + graph_cols
    
    # 组装特征向量
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    
    # 随机森林分类器
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="assembled_features",
        numTrees=cfg["model"]["num_trees"],
        maxDepth=cfg["model"]["max_depth"],
        seed=42,
        featureSubsetStrategy="auto",
        impurity="gini"
    )
    
    # Pipeline
    pipeline = Pipeline(stages=[assembler, rf])
    
    # 划分训练集和测试集（随机划分，因为原始掩码未使用）
    train, test = df.randomSplit([1 - cfg["model"]["test_ratio"], cfg["model"]["test_ratio"]], seed=42)
    print(f"训练集样本数: {train.count()}, 测试集样本数: {test.count()}")
    
    # 训练
    model = pipeline.fit(train)
    
    # 预测
    predictions = model.transform(test)
    
    # 评估指标
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = evaluator_f1.evaluate(predictions)
    
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 保存评估指标到文本文件
    os.makedirs("output/metrics", exist_ok=True)
    with open("output/metrics/model_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"AUC: {auc}\n")
        f.write(f"训练集样本数: {train.count()}\n")
        f.write(f"测试集样本数: {test.count()}\n")
        f.write(f"特征维度: {len(feature_cols)}\n")
        f.write(f"随机森林树数量: {cfg['model']['num_trees']}\n")
        f.write(f"最大深度: {cfg['model']['max_depth']}\n")
    
    # 可选：保存模型
    model_path = "output/model"
    model.write().overwrite().save(model_path)
    print(f"模型已保存至 {model_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()