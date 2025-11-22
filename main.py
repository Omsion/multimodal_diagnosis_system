#!/usr/bin/env python3
"""
多模态DR智能诊断系统主启动脚本
Multimodal Diabetic Retinopathy Diagnosis System Main Launcher

重构后的项目启动入口，兼容原有使用方式。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入并启动FastAPI应用
from src.api.main import app

if __name__ == "__main__":
    import uvicorn

    # 从配置中获取服务器设置
    from src.config.settings import settings

    print(f"🚀 启动 {settings.APP_NAME} v{settings.VERSION}")
    print(f"📡 服务器地址: http://{settings.HOST}:{settings.PORT}")
    print(f"📖 API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG
    )