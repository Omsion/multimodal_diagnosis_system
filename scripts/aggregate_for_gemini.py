#!/usr/bin/env python3
"""
项目代码汇集脚本 - 为Google Gemini API准备完整项目文档

该脚本会递归遍历整个项目目录，生成：
1. 完整的项目目录树形结构
2. 所有Python文件的完整代码内容

生成的文件将包含项目的完整实现，便于输入给Gemini等大语言模型进行分析。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional
import argparse


class ProjectAggregator:
    """项目代码汇集器

    负责收集整个项目的结构信息和代码文件，生成单一的综合文档。
    """

    def __init__(self,
                 root_dir: Optional[Path] = None,
                 output_file: str = "multimodal_dr_diagnosis_for_gemini.txt",
                 include_patterns: Optional[List[str]] = None,
                 exclude_dirs: Optional[List[str]] = None):
        """初始化项目汇集器

        Args:
            root_dir: 项目根目录 (如果为None，自动检测脚本所在目录的父目录作为项目根目录)
            output_file: 输出文件名
            include_patterns: 需要包含的文件模式列表 (默认: ["*.py"])
            exclude_dirs: 需要排除的目录列表
        """
        # 如果没有指定根目录，自动检测项目根目录
        if root_dir is None:
            # 获取脚本所在目录
            script_dir = Path(__file__).parent.resolve()
            # 假设项目根目录是scripts目录的父目录
            self.root_dir = script_dir.parent
        else:
            self.root_dir = Path(root_dir).resolve()
        self.output_file = output_file
        self.include_patterns = include_patterns or ["*.py"]
        self.exclude_dirs = set(exclude_dirs or [
            ".git", "__pycache__", ".idea", ".vscode", "node_modules",
            ".pytest_cache", ".coverage", "htmlcov", "dist", "build"
        ])

        # 统计信息
        self.total_files = 0
        self.total_size = 0
        self.processed_files = []

        print(f"项目根目录: {self.root_dir}")
        print(f"输出文件: {self.output_file}")

    def is_excluded(self, path: Path) -> bool:
        """检查路径是否应该被排除

        Args:
            path: 要检查的路径

        Returns:
            bool: True表示应该排除
        """
        # 检查是否在排除的目录中
        for part in path.parts:
            if part in self.exclude_dirs:
                return True
        return False

    def should_include_file(self, file_path: Path) -> bool:
        """检查文件是否应该被包含

        Args:
            file_path: 文件路径

        Returns:
            bool: True表示应该包含
        """
        if self.is_excluded(file_path):
            return False

        # 排除脚本自身
        if file_path.resolve() == Path(__file__).resolve():
            return False

        # 检查文件扩展名
        for pattern in self.include_patterns:
            if file_path.match(pattern):
                return True
        return False

    def generate_tree_structure(self, max_depth: int = 3) -> str:
        """生成项目目录树形结构

        Args:
            max_depth: 显示的最大深度

        Returns:
            str: 格式化的树形结构字符串
        """
        tree_lines = []
        tree_lines.append("📁 项目目录结构:")
        tree_lines.append("=" * 60)

        def _build_tree(directory: Path, prefix: str = "", depth: int = 0) -> None:
            """递归构建树形结构

            Args:
                directory: 当前目录
                prefix: 前缀字符串
                depth: 当前深度
            """
            if depth > max_depth:
                tree_lines.append(f"{prefix}... (最大深度 {max_depth})")
                return

            if self.is_excluded(directory):
                return

            try:
                # 获取目录内容并排序
                items = sorted([item for item in directory.iterdir()
                              if not self.is_excluded(item)],
                             key=lambda x: (x.is_file(), x.name.lower()))

                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "

                    if item.is_dir():
                        tree_lines.append(f"{prefix}{current_prefix}📁 {item.name}/")
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        _build_tree(item, next_prefix, depth + 1)
                    else:
                        # 显示文件图标
                        icon = self._get_file_icon(item)
                        tree_lines.append(f"{prefix}{current_prefix}{icon} {item.name}")

            except PermissionError:
                tree_lines.append(f"{prefix}└── [权限不足]")

        _build_tree(self.root_dir)
        return "\n".join(tree_lines)

    def _get_file_icon(self, file_path: Path) -> str:
        """获取文件对应的图标

        Args:
            file_path: 文件路径

        Returns:
            str: 文件图标emoji
        """
        suffix = file_path.suffix.lower()
        icon_map = {
            ".py": "🐍",
            ".js": "🟨",
            ".ts": "🔷",
            ".json": "📋",
            ".yaml": "📄",
            ".yml": "📄",
            ".md": "📝",
            ".txt": "📄",
            ".csv": "📊",
            ".png": "🖼️",
            ".jpg": "🖼️",
            ".jpeg": "🖼️",
            ".gif": "🖼️",
            ".pdf": "📕",
            ".html": "🌐",
            ".css": "🎨",
            ".sql": "🗃️",
            ".sh": "💻",
            ".bat": "💻",
            ".ps1": "💻",
        }
        return icon_map.get(suffix, "📄")

    def collect_python_files(self) -> List[Path]:
        """收集所有Python文件

        Returns:
            List[Path]: Python文件路径列表
        """
        python_files = []

        print("搜索Python文件...")
        for root, dirs, files in os.walk(self.root_dir):
            # 过滤掉排除的目录
            dirs[:] = [d for d in dirs if not self.is_excluded(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file
                if self.should_include_file(file_path):
                    python_files.append(file_path)

        print(f"找到 {len(python_files)} 个Python文件")
        return sorted(python_files)

    def format_file_content(self, file_path: Path) -> str:
        """格式化文件内容

        Args:
            file_path: 文件路径

        Returns:
            str: 格式化后的文件内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return f"// 无法读取文件 {file_path}: 编码错误"
        except Exception as e:
            return f"// 读取文件 {file_path} 时出错: {str(e)}"

        # 获取相对路径
        rel_path = file_path.relative_to(self.root_dir)

        # 构建文件头部
        header = [
            "=" * 80,
            f"📁 文件路径: {rel_path}",
            f"📏 文件大小: {len(content)} 字节",
            f"🕒 最后修改: {datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]

        # 如果文件为空，添加提示
        if not content.strip():
            content = "# 此文件为空"

        return "\n".join(header) + content + "\n\n"

    def generate_summary(self, python_files: List[Path]) -> str:
        """生成项目统计摘要

        Args:
            python_files: Python文件列表

        Returns:
            str: 格式化的统计摘要
        """
        total_lines = 0
        total_chars = 0
        file_stats = []

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                lines = len(content.splitlines())
                chars = len(content)
                total_lines += lines
                total_chars += chars
                file_stats.append({
                    'path': file_path.relative_to(self.root_dir),
                    'lines': lines,
                    'chars': chars
                })
            except:
                pass

        summary = [
            "📊 项目统计信息:",
            "=" * 60,
            f"🐍 Python文件总数: {len(python_files)}",
            f"📝 总代码行数: {total_lines:,}",
            f"💾 总字符数: {total_chars:,}",
            f"📈 平均每文件行数: {total_lines // len(python_files) if python_files else 0}",
            ""
        ]

        # 添加最大的10个文件
        if file_stats:
            summary.append("📋 最大的10个Python文件:")
            file_stats.sort(key=lambda x: x['lines'], reverse=True)
            for i, stat in enumerate(file_stats[:10], 1):
                summary.append(f"  {i:2d}. {stat['path']} ({stat['lines']} 行)")

        summary.append("")
        return "\n".join(summary)

    def run(self) -> None:
        """执行项目汇集任务"""
        print("开始汇集项目代码...")
        start_time = datetime.now()

        # 创建输出目录
        output_path = self.root_dir / self.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 收集内容
        content_sections = []

        # 1. 添加头部信息和Gemini引导语
        header = [
            "#" * 80,
            "# 多模态医学影像诊断系统 - 完整项目代码",
            "#" * 80,
            f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# 项目根目录: {self.root_dir}",
            f"# 输出文件: {self.output_file}",
            f"# 目标模型: Google Gemini",
            "#" * 80,
            ""
        ]
        content_sections.append("\n".join(header))

        # 添加Gemini引导语
        gemini_prompt = self._generate_gemini_prompt()
        content_sections.append(gemini_prompt)

        # 2. 生成项目目录树
        print("生成项目目录结构...")
        content_sections.append(self.generate_tree_structure())
        content_sections.append("\n\n")

        # 3. 收集并格式化所有Python文件
        python_files = self.collect_python_files()
        content_sections.append(self.generate_summary(python_files))

        print("读取Python文件内容...")
        content_sections.append("\n" + "=" * 80 + "\n")
        content_sections.append("Python代码文件详细内容:\n")

        for i, file_path in enumerate(python_files, 1):
            print(f"  ({i}/{len(python_files)}) 处理: {file_path.relative_to(self.root_dir)}")
            content_sections.append(self.format_file_content(file_path))
            self.processed_files.append(file_path)

        # 4. 添加结束标记
        footer = [
            "=" * 80,
            "# 项目代码汇集完成",
            f"# 统计: {len(python_files)} 个Python文件",
            f"# 用时: {(datetime.now() - start_time).total_seconds():.2f} 秒",
            "=" * 80
        ]
        content_sections.append("\n".join(footer))

        # 写入输出文件
        print(f"写入输出文件: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content_sections))

        # 完成信息
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("项目代码汇集完成!")
        print(f"输出文件: {output_path}")
        print(f"处理文件数: {len(python_files)}")
        print(f"输出文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"总用时: {duration:.2f} 秒")

    def _generate_gemini_prompt(self) -> str:
        """生成针对Gemini的引导语

        Returns:
            str: Gemini引导语
        """
        return """
🤖 Gemini AI 引导语:

Hello Gemini! I need your expert help with my Python project for a multi-modal medical diagnosis system.

📋 Your Role:
Act as an expert AI engineer and Python developer with deep specialization in:
- Multi-modal systems (vision + language)
- Computer vision and medical image analysis
- Natural language processing and RAG systems
- LangChain and FastAPI framework development
- PyTorch and transformer models

🏥 Project Context:
The code I'm providing is a complete system for **Multi-Modal Diabetic Retinopathy (DR) Diagnosis**.
It integrates computer vision with Large Language Models to provide intelligent medical diagnosis.

🏗️ System Architecture:
1. **Configuration Management** (`settings.py`): Centralized Pydantic settings for model paths and parameters
2. **Vision Processing** (`vision_processors.py`):
   - ResNet50 model for DR grading and classification
   - Qwen-VL visual language model for generating lesion descriptions
3. **LLM Integration** (`llm_loader.py`): R1-7B model with LoRA fine-tuning, LangChain compatible
4. **RAG Pipeline** (`rag_chain_builder.py`): Advanced retrieval-augmented generation using:
   - FAISS vector store for medical knowledge retrieval
   - LangChain Expression Language (LCEL) for chain construction
   - Chain-of-Thought reasoning prompts
5. **FastAPI Service** (`main.py`): RESTful API with `/diagnose` endpoint for complete workflow
6. **Tools & Utilities** (`utils/`): Helper functions for data processing and system operations

🎯 Your Task:
1. **Analyze & Understand**: Comprehend the entire codebase, understanding component interactions
2. **Confirm Understanding**: Respond with "I have analyzed the complete multi-modal DR diagnosis system and understand the workflow from visual analysis to RAG-based diagnostic report generation. I am ready to assist. What would you like me to help with?"
3. **Provide Assistance**: Help with debugging, code improvements, architecture suggestions, feature additions, and optimization

📁 Code Structure:
All Python files are provided below with clear file path delimiters and content formatting.

"""


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="汇集项目代码到单一文件，便于输入给Gemini等大语言模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python aggregate_for_gemini.py
  python aggregate_for_gemini.py --output my_project.txt
  python aggregate_for_gemini.py --include "*.py" "*.yaml" "*.md"
        """
    )

    parser.add_argument(
        "--output", "-o",
        default="multimodal_dr_diagnosis_for_gemini.txt",
        help="输出文件名 (默认: multimodal_dr_diagnosis_for_gemini.txt)"
    )

    parser.add_argument(
        "--include", "-i",
        nargs="*",
        default=["*.py"],
        help="要包含的文件模式 (默认: ['*.py'])"
    )

    parser.add_argument(
        "--exclude", "-e",
        nargs="*",
        default=[".git", "__pycache__", ".idea", ".vscode", "node_modules"],
        help="要排除的目录 (默认: ['.git', '__pycache__', '.idea', '.vscode', 'node_modules'])"
    )

    parser.add_argument(
        "--root", "-r",
        default=None,
        help="项目根目录 (默认: 自动检测项目根目录)"
    )

    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=5,
        help="目录树显示的最大深度 (默认: 5)"
    )

    args = parser.parse_args()

    # 检查根目录是否存在（如果指定了的话）
    root_dir = None
    if args.root:
        root_dir = Path(args.root)
        if not root_dir.exists():
            print(f"错误: 根目录不存在: {root_dir}")
            sys.exit(1)

    # 创建并运行汇集器
    aggregator = ProjectAggregator(
        root_dir=root_dir,
        output_file=args.output,
        include_patterns=args.include,
        exclude_dirs=args.exclude
    )

    try:
        aggregator.run()
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"运行时错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()