
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from datetime import datetime

from src.core.vision.processors import DRGradingModule
from src.config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dr_grading_service")

app = FastAPI(
    title="DR Grading Service",
    description="专门负责DR分级的微服务",
    version="1.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
dr_grader = None

@app.on_event("startup")
async def startup_event():
    global dr_grader
    logger.info("正在加载DR分级模型...")
    try:
        dr_grader = DRGradingModule(settings.RESNET_MODEL_PATH)
        # 预加载模型
        dr_grader._load_model()
        logger.info("DR分级模型加载成功")
    except Exception as e:
        logger.error(f"DR分级模型加载失败: {e}")
        # 不抛出异常，允许服务启动，但在调用时报错

@app.post("/predict_dr")
async def predict_dr(file: UploadFile = File(...)):
    global dr_grader
    if not dr_grader:
        raise HTTPException(status_code=503, detail="模型未初始化")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 预测
        grade, confidence = dr_grader.predict(image)
        
        return {
            "dr_grade": grade,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": dr_grader is not None and dr_grader._model_loaded}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
