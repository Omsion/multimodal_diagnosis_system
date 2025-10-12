# rag_chain_builder.py
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from settings import settings
import logging

logger = logging.getLogger(__name__)

def get_retriever():
    """创建并返回一个知识库检索器。"""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    if os.path.exists(settings.VECTOR_DB_PATH):
        vector_store = FAISS.load_local(settings.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("从本地加载向量数据库。")
    else:
        loader = DirectoryLoader(settings.KNOWLEDGE_BASE_PATH, glob="**/*.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(settings.VECTOR_DB_PATH)
        logger.info("创建并保存了新的向量数据库。")
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
    
    ### 思考过程:
    让我一步步思考：
    1.  **病灶与分级关联分析**: 病灶描述 '{lesion_description}' 与诊断 '{dr_grade_desc}' 是否一致？知识库中对此有何说明？
    2.  **风险评估**: 根据分级和病灶，患者的视力丧失风险有多大？是否有进展迹象？
    3.  **治疗方案选择**: 知识库中针对 '{dr_grade_desc}' 推荐了哪些治疗方案？结合具体病灶，哪种最合适？
    
    ### 结构化输出:
    {format_instructions}
    """
    
    format_instructions = """
    请严格按照以下JSON格式输出，不要包含任何代码块标记或额外说明：
    {
      "cot_reasoning": "（这里是你详细的思考过程）",
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