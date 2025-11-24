# rag_chain_builder.py
"""
RAG (Retrieval-Augmented Generation) 链构建器

该模块负责创建和管理检索增强生成链，用于糖尿病视网膜病变诊断的知识增强推理。
包含向量数据库管理、文档检索、提示模板构建等功能。
"""

import os
import logging
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 使用新的langchain-huggingface包
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import settings

logger = logging.getLogger(__name__)

def get_retriever():
    """创建并返回一个知识库检索器。"""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    if os.path.exists(settings.VECTOR_DB_PATH):
        vector_store = FAISS.load_local(settings.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("从本地加载向量数据库。")
    else:
        # 直接读取文本文件，避免复杂的DocumentLoader问题
        documents = []
        knowledge_file = os.path.join(settings.KNOWLEDGE_BASE_PATH, "dr_treatment_guidelines.txt")

        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 创建简单的Document对象
                from langchain.docstore.document import Document
                doc = Document(page_content=content, metadata={"source": knowledge_file})
                documents.append(doc)
        else:
            logger.warning(f"知识库文件不存在: {knowledge_file}")
            # 创建默认知识库
            default_content = """
            # 糖尿病视网膜病变治疗指南

            ## 轻度非增殖性DR (Mild NPDR)
            - 治疗: 控制血糖、血压、血脂
            - 随访: 每6-12个月复查一次

            ## 中度非增殖性DR (Moderate NPDR)
            - 治疗: 严格控制全身情况，每3-6个月复查
            - 可能需要激光治疗

            ## 重度非增殖性DR (Severe NPDR)
            - 治疗: 考虑进行全视网膜光凝治疗
            - 随访: 每2-4个月密切随访

            ## 增殖性DR (PDR)
            - 治疗: 立即进行抗VEGF治疗或激光治疗
            - 随访: 每月或更频繁随访
            """
            from langchain.docstore.document import Document
            doc = Document(page_content=default_content, metadata={"source": "default"})
# rag_chain_builder.py
"""
RAG (Retrieval-Augmented Generation) 链构建器

该模块负责创建和管理检索增强生成链，用于糖尿病视网膜病变诊断的知识增强推理。
包含向量数据库管理、文档检索、提示模板构建等功能。
"""

import os
import logging
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 使用新的langchain-huggingface包
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import settings

logger = logging.getLogger(__name__)

def get_retriever():
    """创建并返回一个知识库检索器。"""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    if os.path.exists(settings.VECTOR_DB_PATH):
        vector_store = FAISS.load_local(settings.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("从本地加载向量数据库。")
    else:
        # 直接读取文本文件，避免复杂的DocumentLoader问题
        documents = []
        knowledge_file = os.path.join(settings.KNOWLEDGE_BASE_PATH, "dr_treatment_guidelines.txt")

        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 创建简单的Document对象
                from langchain.docstore.document import Document
                doc = Document(page_content=content, metadata={"source": knowledge_file})
                documents.append(doc)
        else:
            logger.warning(f"知识库文件不存在: {knowledge_file}")
            # 创建默认知识库
            default_content = """
            # 糖尿病视网膜病变治疗指南

            ## 轻度非增殖性DR (Mild NPDR)
            - 治疗: 控制血糖、血压、血脂
            - 随访: 每6-12个月复查一次

            ## 中度非增殖性DR (Moderate NPDR)
            - 治疗: 严格控制全身情况，每3-6个月复查
            - 可能需要激光治疗

            ## 重度非增殖性DR (Severe NPDR)
            - 治疗: 考虑进行全视网膜光凝治疗
            - 随访: 每2-4个月密切随访

            ## 增殖性DR (PDR)
            - 治疗: 立即进行抗VEGF治疗或激光治疗
            - 随访: 每月或更频繁随访
            """
            from langchain.docstore.document import Document
            doc = Document(page_content=default_content, metadata={"source": "default"})
            documents.append(doc)

        # 分块处理
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)

        # 创建向量数据库
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(settings.VECTOR_DB_PATH)
        logger.info(f"创建并保存了新的向量数据库，共处理 {len(docs)} 个文档块。")
    return vector_store.as_retriever(search_kwargs={"k": settings.TOP_K})

def create_rag_chain(llm: BaseChatModel, retriever):
    """使用LCEL构建并返回RAG链。"""
    prompt_template = """
    ## 角色定位
    你是一名专业的眼科医生AI助手。

    ## 核心任务
    根据提供的`DR分级结果`、`关键病灶描述`和`参考知识`，生成一份结构化、可解释、可追溯的诊疗决策报告。
    你必须严格遵循思维链（Chain-of-Thought）进行推理。

    ## 参考知识
    {context}

    ## 诊断信息
    - DR分级结果: {dr_grade_desc}
    - 关键病灶描述: {lesion_description}

    ## 推理与决策
    请基于以上信息，进行链式思维(Chain-of-Thought)推理，并生成最终的诊疗决策。
    
    请严格按照以下格式输出：
    
    Thinking Process:
    (这里是你详细的思考过程，请一步步分析：
    1. 病灶与分级关联分析
    2. 风险评估
    3. 治疗方案选择)
    
    JSON Report:
    {format_instructions}
    """
    
    format_instructions = """
    {
      "recommendations": [
        "（具体的治疗建议1）",
        "（具体的随访计划建议2）"
      ],
      "traceability": "（引用支持你决策的`参考知识`中的关键句子）"
    }
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "dr_grade_desc", "lesion_description"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # 定义输入到RAG链的数据结构
    setup = RunnableParallel(
        context=lambda x: retriever.get_relevant_documents(x["lesion_description"]),
        dr_grade_desc=lambda x: x["dr_grade_desc"],
        lesion_description=lambda x: x["lesion_description"]
    )
    
    rag_chain = setup | prompt | llm | StrOutputParser()
    logger.info("LangChain RAG链构建完成。")
    return rag_chain