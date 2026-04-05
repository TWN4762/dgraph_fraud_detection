#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os

def load_config(config_path="conf/config.yaml"):
    """
    加载 YAML 配置文件，并返回配置字典。
    如果文件不存在，抛出异常。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 可以在这里添加一些默认值或检查
    return config

# 测试
if __name__ == "__main__":
    cfg = load_config()
    print("配置加载成功:")
    for key in cfg:
        print(f"  {key}: {cfg[key]}")