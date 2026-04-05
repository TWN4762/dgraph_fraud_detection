#!/bin/bash
# 02_run_spark_analysis.sh
# 依次执行图构建、特征合并、模型训练、可视化

cd /home/wushihang/dgraph_fraud_detection

echo "=== Step 1: 图构建 ==="
spark-submit \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.catalogImplementation=hive \
  --driver-memory 4g \
  src/02_graph_build.py

echo "=== Step 2: 特征合并 ==="
spark-submit \
  --conf spark.sql.catalogImplementation=hive \
  --driver-memory 4g \
  src/03_feature_merge.py

echo "=== Step 3: 模型训练与评估 ==="
spark-submit \
  --conf spark.sql.catalogImplementation=hive \
  --driver-memory 4g \
  src/04_model_train.py

echo "=== Step 4: 可视化 ==="
spark-submit \
  --conf spark.sql.catalogImplementation=hive \
  --driver-memory 2g \
  src/05_visualize.py

echo "所有分析任务完成！"