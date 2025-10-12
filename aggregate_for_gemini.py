import os
import sys

# --- 配置 ---

# 最终输出的txt文件名
OUTPUT_FILENAME = "multimodal_dr_diagnosis_for_gemini.txt"

# 需要从聚合中排除的目录和文件。
# 这可以防止包含虚拟环境、数据、模型、Git历史等无关内容。
EXCLUDED_DIRS = {
    '__pycache__',
    '.git',
    '.idea',
    '.vscode',
    'venv',
    '.venv',
    'knowledge_base',  # 包含私有知识，不是源代码
    'models',  # 包含大型模型文件，不是源代码
    'vector_db',  # 包含生成的向量数据库，不是源代码
}
EXCLUDED_FILES = {
    'aggregate_for_gemini.py',  # 排除此脚本自身
    '.env',  # 排除敏感配置
}

# 引导语 (Prompt)，为AI提供精准的项目背景和任务指令。
# 这是本脚本最核心的部分，需要精准描述项目结构和目标。
GEMINI_PROMPT = """
Hello Gemini. I need your expert help with my Python project for a multi-modal medical diagnosis system.

Your Role:
Act as an expert AI engineer and Python developer with deep specialization in multi-modal systems, computer vision, natural language processing (NLP), and the implementation of Retrieval-Augmented Generation (RAG) pipelines using LangChain and FastAPI.

Project Context:
The code I'm providing is a complete system for **Multi-Modal Diabetic Retinopathy (DR) Diagnosis**. It's designed to simulate a clinical diagnostic workflow by integrating computer vision models with a Large Language Model (LLM) powered by a private knowledge base.

The project follows a sophisticated, decoupled architecture:
1.  **Configuration (`settings.py`):** Centralized Pydantic settings manage all model paths and parameters, allowing easy configuration via a `.env` file.
2.  **Vision Processing (`vision_processors.py`):** This module is responsible for all visual tasks. It contains:
    *   A fine-tuned **ResNet50** model for DR grading (classifying the severity).
    *   A Visual Language Model (**Qwen-VL**) to generate descriptive text about key lesions from the fundus image.
3.  **LLM Loading (`llm_loader.py`):** Loads a fine-tuned Large Language Model (**R1-7B with LoRA**) and wraps it into a standard LangChain-compatible component for seamless integration.
4.  **RAG Pipeline (`rag_chain_builder.py`):** This is the core of the NLP logic. It uses **LangChain Expression Language (LCEL)** to build a RAG chain that:
    *   Loads private medical guidelines from the `knowledge_base/` directory.
    *   Creates a vector store using FAISS for efficient retrieval.
    *   Defines a sophisticated prompt template that guides the LLM to perform Chain-of-Thought (CoT) reasoning.
5.  **API Service (`main.py`):** A **FastAPI** application serves as the central controller. It exposes a single `/diagnose` endpoint that:
    *   Receives an uploaded fundus image.
    *   Orchestrates the calls to the vision processing modules (ResNet50 and Qwen-VL).
    *   Invokes the RAG chain with the results from the vision models.
    *   Returns a structured, traceable, and clinically relevant diagnostic report in JSON format.
6.  **Client Script (`run_diagnosis.py`):** A command-line tool to easily test the entire system by sending an image to the FastAPI server and printing the final report.

Your Task:

1.  **Analyze and Understand:** Carefully read and fully comprehend the entire Python codebase provided below. The code is split into multiple files, and you must understand how they interact. The main entry point for the service is `main.py`, and for testing is `run_diagnosis.py`.

2.  **Wait for Instructions:** After you have fully processed all the code, simply respond with: "I have analyzed the complete multi-modal DR diagnosis system. I understand the workflow, from visual analysis to the RAG-based generation of traceable diagnostic reports. I am ready to assist. What is your question?"

3.  **Assist Me:** Once you've given the confirmation message, I will ask you questions. You should then help me with tasks such as:
    *   Debugging specific errors in any part of the pipeline.
    *   Explaining complex parts of the code (e.g., the LCEL chain construction in `rag_chain_builder.py`).
    *   Suggesting code improvements for better performance, async handling, or modularity.
    *   Refactoring the code to follow different design patterns.
    *   Adding new features, such as caching mechanisms for the RAG retriever or integrating different VLM models.

Code Structure:
The complete source code is provided below. Each file is clearly delimited by `--- START OF FILE: [filepath] ---` and `--- END OF FILE: [filepath] ---` markers.
"""


def aggregate_scripts():
    """
    递归地查找项目目录下的所有.py文件，并将它们的内容
    合并到一个带有引导性Prompt的txt文件中。
    """
    try:
        # 假设此脚本位于项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))

        print(f"项目根目录: {project_root}")
        print(f"开始聚合.py文件...")

        py_files_to_aggregate = []
        for root, dirs, files in os.walk(project_root, topdown=True):
            # 修改dirs列表可以阻止os.walk深入到被排除的目录中
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file in files:
                if file.endswith('.py') and file not in EXCLUDED_FILES:
                    # 获取相对于项目根的路径用于在输出文件中标记，更清晰
                    relative_path = os.path.relpath(os.path.join(root, file), project_root)
                    py_files_to_aggregate.append(relative_path)

        # 排序以确保每次运行生成的文件内容顺序一致
        py_files_to_aggregate.sort()

        if not py_files_to_aggregate:
            print("错误: 未找到任何可供聚合的.py文件。请确保此脚本位于项目根目录下且未被排除。")
            return

        print(f"\n即将聚合以下 {len(py_files_to_aggregate)} 个文件:")
        for rel_path in py_files_to_aggregate:
            print(f"- {rel_path}")

        # 开始写入输出文件
        output_filepath = os.path.join(project_root, OUTPUT_FILENAME)
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # 1. 写入引导语
            outfile.write(GEMINI_PROMPT)

            # 2. 在引导语后附上被聚合的文件列表，使其更清晰
            outfile.write("\n\n**Aggregated Files:**\n")
            for rel_path in py_files_to_aggregate:
                outfile.write(f"- `{rel_path.replace(os.sep, '/')}`\n")
            outfile.write("\n" + "=" * 80 + "\n")
            outfile.write("--- START OF AGGREGATED CODE ---\n" + "=" * 80 + "\n\n")

            # 3. 依次读取每个文件并写入
            for rel_path in py_files_to_aggregate:
                absolute_path = os.path.join(project_root, rel_path)
                try:
                    with open(absolute_path, 'r', encoding='utf-8') as infile:
                        # 使用相对路径和正斜杠作为文件标记
                        clean_rel_path = rel_path.replace(os.sep, '/')
                        outfile.write(f"--- START OF FILE: {clean_rel_path} ---\n\n")
                        outfile.write(infile.read())
                        outfile.write(f"\n\n--- END OF FILE: {clean_rel_path} ---\n\n\n")
                except Exception as e:
                    error_message = f"--- ERROR READING FILE: {rel_path} ---\n"
                    error_message += f"--- REASON: {str(e)} ---\n\n"
                    outfile.write(error_message)
                    print(f"\n警告: 读取文件 {rel_path} 时发生错误: {e}")

        print(f"\n✅ **成功!** 已将 {len(py_files_to_aggregate)} 个脚本合并到: {output_filepath}")
        print("您现在可以将此文件的内容粘贴到 Gemini 中。")

    except Exception as e:
        print(f"\n❌ **发生严重错误:** {e}")
        sys.exit(1)


if __name__ == "__main__":
    aggregate_scripts()