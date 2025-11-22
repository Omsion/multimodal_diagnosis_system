#!/usr/bin/env python3
"""
向量数据库初始化脚本

该脚本用于从治疗知识库创建FAISS向量数据库，
确保RAG系统能够正确检索DR治疗信息。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings
from src.core.rag.chain_builder import get_retriever
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """初始化向量数据库"""
    print("=" * 60)
    print("初始化DR治疗知识库向量数据库")
    print("=" * 60)

    # 检查知识库文件是否存在
    knowledge_file = Path(settings.KNOWLEDGE_BASE_PATH) / "dr_treatment_guidelines.txt"
    if not knowledge_file.exists():
        logger.error(f"知识库文件不存在: {knowledge_file}")
        print(f"\n请确保以下文件存在:")
        print(f"  - {knowledge_file}")
        return False

    print(f"知识库文件: {knowledge_file}")
    print(f"文件大小: {knowledge_file.stat().st_size / 1024:.1f} KB")

    try:
        # 删除旧的向量数据库（如果存在）
        if os.path.exists(settings.VECTOR_DB_PATH):
            import shutil
            shutil.rmtree(settings.VECTOR_DB_PATH)
            print(f"删除旧的向量数据库: {settings.VECTOR_DB_PATH}")

        # 创建检索器（会自动创建向量数据库）
        print(f"\n正在创建向量数据库...")
        print(f"存储路径: {settings.VECTOR_DB_PATH}")
        print(f"嵌入模型: {settings.EMBEDDING_MODEL}")
        print(f"分块大小: {settings.CHUNK_SIZE} 字符")
        print(f"检索数量: {settings.TOP_K} 个文档")

        retriever = get_retriever()

        print(f"\n向量数据库创建成功!")
        print(f"数据库路径: {settings.VECTOR_DB_PATH}")

        # 测试检索功能
        print(f"\n测试检索功能...")
        test_queries = [
            "轻度非增殖性DR的治疗方案",
            "增殖性DR的激光治疗",
            "黄斑水肿的抗VEGF治疗",
            "糖尿病视网膜病变随访建议"
        ]

        for query in test_queries:
            docs = retriever.get_relevant_documents(query)
            print(f"查询: '{query}'")
            print(f"   找到 {len(docs)} 个相关文档")
            if docs:
                print(f"   第一个文档片段: {docs[0].page_content[:100]}...")
            print()

        print("向量数据库初始化完成！RAG系统已准备就绪。")
        return True

    except Exception as e:
        logger.error(f"向量数据库创建失败: {e}")
        print(f"\n错误详情: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)