"""代码审查Agent测试配置 — 添加项目路径到系统路径"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))