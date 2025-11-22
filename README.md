# 多模态糖尿病视网膜病变诊断系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

一个结合计算机视觉、大型语言模型和检索增强生成(RAG)技术的智能糖尿病视网膜病变(DR)诊断系统。

## 🎯 系统功能

- 🏥 **DR严重程度自动分级**: 基于ResNet50的5级分类
- 📝 **病灶特征智能描述**: 使用Qwen-VL视觉语言模型
- 🧠 **医学知识推理**: 集成R1-7B医学LLM进行CoT推理
- 📚 **知识库增强**: 基于FAISS的向量检索系统
- 🌐 **Web界面**: 完整的前后端交互界面
- 🔄 **实时监控**: 系统健康状态和日志监控

## 📁 项目结构

```
multimodal_diagnosis_system/
├── src/                          # 核心源代码
│   ├── api/                      # FastAPI服务层
│   │   └── main.py              # 主服务文件
│   ├── core/                     # 核心业务逻辑
│   │   ├── vision/              # 视觉处理模块
│   │   ├── llm/                 # LLM集成模块
│   │   └── rag/                 # RAG系统模块
│   ├── config/                   # 配置管理
│   │   └── settings.py          # 系统配置
│   └── utils/                    # 工具函数
├── scripts/                      # 运行脚本
│   ├── run_diagnosis.py         # 单图诊断
│   ├── test_system.py           # 系统测试
│   └── deploy.py                # 部署脚本
├── data/                         # 数据文件
│   ├── knowledge_base/          # 医学知识库
│   └── processed/               # 处理后数据
├── web/                          # Web前端
│   └── frontend.html
├── models/                       # 预训练模型
├── medical-o1-reasoning-SFT/     # SFT训练数据
├── docs/                         # 项目文档
├── tests/                        # 测试文件
├── requirements.txt
├── .env.example
└── main.py                      # 启动入口
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd multimodal_diagnosis_system

# 创建虚拟环境
conda create -n dr-diagnosis python=3.8
conda activate dr-diagnosis

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件，设置模型路径等参数
# 确保以下路径存在：
# - ./models/resnet50_dr_grading.pth
# - ./models/Qwen-VL/
# - ./models/R1-7B-finetuned/
```

### 3. 启动服务

```bash
# 启动FastAPI服务
python main.py
# 或者
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问系统

- 🌐 **Web界面**: http://localhost:8000/web/frontend.html
- 📖 **API文档**: http://localhost:8000/docs
- 🔍 **健康检查**: http://localhost:8000/health

## 📖 使用指南

### 单图诊断

```bash
# 使用脚本进行单图诊断
python scripts/run_diagnosis.py path/to/image.jpg

# 或通过API调用
curl -X POST "http://localhost:8000/diagnose" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"
```

### 系统测试

```bash
# 运行完整系统测试
python scripts/test_system.py

# 部署测试
python scripts/deploy.py
```

## ⚙️ 配置说明

主要配置文件位于 `src/config/settings.py`：

- **模型配置**: 各模型路径和设备设置
- **API配置**: 服务器端口、CORS设置等
- **RAG配置**: 知识库路径、检索参数
- **日志配置**: 日志级别和输出格式

## 🧠 技术架构

### 核心技术栈

- **后端框架**: FastAPI + Pydantic
- **深度学习**: PyTorch + Transformers + torchvision
- **LLM集成**: LangChain + R1-7B
- **向量检索**: FAISS + sentence-transformers
- **前端**: HTML5 + Tailwind CSS

### 系统流程

1. **图像输入** → 上传眼底图像
2. **视觉处理** → ResNet50分级 + Qwen-VL描述 (并行)
3. **知识检索** → FAISS检索相关医学知识
4. **推理生成** → R1-7B进行CoT推理和报告生成
5. **结果输出** → 结构化JSON诊断报告

## 🔧 开发指南

### 添加新功能

1. 在对应模块下创建新文件
2. 更新 `__init__.py` 导出接口
3. 添加相应的测试用例
4. 更新配置文件(如需要)

### 代码规范

- 使用类型提示 (Type Hints)
- 遵循 Google 风格的文档字符串
- 使用 `pathlib.Path` 处理文件路径
- 保持代码简洁，避免过度工程化

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 贡献

欢迎提交 Issue 和 Pull Request 来改进系统！

## 📞 联系

如有问题，请通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

**注意**: 本系统仅用于研究和教育目的，不能替代专业医疗诊断。