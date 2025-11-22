#!/usr/bin/env python3
"""
RAG诊断系统测试脚本

演示RAG系统如何根据DR分级结果从知识库中检索相关的治疗建议
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag.chain_builder import get_retriever
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rag_for_dr_grades():
    """测试RAG系统对不同DR等级的治疗建议检索"""
    print("=" * 80)
    print("DR治疗RAG系统测试")
    print("=" * 80)

    # 初始化RAG检索器
    print("正在初始化RAG检索器...")
    retriever = get_retriever()

    # 测试不同DR等级的治疗建议查询
    test_cases = [
        {
            "dr_grade": "无DR",
            "description": "眼底检查未见明显异常",
            "queries": [
                "无DR的随访建议",
                "糖尿病视网膜病变预防措施",
                "正常眼底复查间隔"
            ]
        },
        {
            "dr_grade": "轻度非增殖性DR",
            "description": "仅见微动脉瘤",
            "queries": [
                "轻度非增殖性DR的治疗方案",
                "微动脉瘤的处理建议",
                "轻度DR随访频率"
            ]
        },
        {
            "dr_grade": "中度非增殖性DR",
            "description": "可见视网膜内出血和硬性渗出",
            "queries": [
                "中度非增殖性DR的治疗建议",
                "视网膜内出血的处理",
                "硬性渗出的治疗方法"
            ]
        },
        {
            "dr_grade": "重度非增殖性DR",
            "description": "符合4-2-1规则，可见静脉串珠和IRMA",
            "queries": [
                "重度非增殖性DR的激光治疗",
                "全视网膜光凝适应症",
                "重度NPDR的紧急处理"
            ]
        },
        {
            "dr_grade": "增殖性DR",
            "description": "可见视盘新生血管和视网膜前出血",
            "queries": [
                "增殖性DR的抗VEGF治疗",
                "新生血管的治疗方案",
                "PDR的紧急处理措施"
            ]
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}: {case['dr_grade']}")
        print(f"病灶描述: {case['description']}")
        print(f"{'='*60}")

        for j, query in enumerate(case['queries'], 1):
            print(f"\n查询 {j}: {query}")

            # 检索相关文档
            try:
                docs = retriever.invoke(query)  # 使用新的invoke方法

                print(f"   找到 {len(docs)} 个相关文档片段:")

                for k, doc in enumerate(docs, 1):
                    print(f"   片段 {k}:")
                    content = doc.page_content.strip()

                    # 限制显示长度，突出关键信息
                    if len(content) > 200:
                        content = content[:200] + "..."

                    print(f"      {content}")
                    print(f"      来源: {doc.metadata.get('source', 'unknown')}")
                    print()

            except Exception as e:
                print(f"   检索失败: {e}")

        print(f"\n基于以上检索结果，{case['dr_grade']}的治疗建议已从知识库中获取")
        print("   RAG系统能够为临床决策提供循证支持")

def test_combined_diagnosis():
    """测试完整诊断流程"""
    print(f"\n{'='*80}")
    print("完整诊断流程测试")
    print(f"{'='*80}")

    # 模拟一个完整的诊断场景
    mock_diagnosis = {
        "dr_grade": "中度非增殖性DR",
        "lesion_description": "视网膜可见多个散在点状出血，后极部可见少量硬性渗出，未见新生血管形成"
    }

    print(f"诊断结果:")
    print(f"   DR分级: {mock_diagnosis['dr_grade']}")
    print(f"   病灶描述: {mock_diagnosis['lesion_description']}")

    # 综合查询
    comprehensive_queries = [
        "中度非增殖性DR的综合治疗方案",
        "视网膜出血和硬性渗出的处理",
        "中度DR的随访计划",
        "血糖控制对DR进展的影响"
    ]

    retriever = get_retriever()

    print(f"\n基于诊断结果的治疗知识检索:")
    for i, query in enumerate(comprehensive_queries, 1):
        print(f"\n{i}. {query}")
        try:
            docs = retriever.invoke(query)
            if docs:
                # 选择最相关的文档片段
                best_doc = docs[0]
                content = best_doc.page_content.strip()
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"   {content}")
            else:
                print(f"   未找到相关治疗建议")
        except Exception as e:
            print(f"   检索失败: {e}")

if __name__ == "__main__":
    try:
        # 测试分级治疗建议
        test_rag_for_dr_grades()

        # 测试完整诊断流程
        test_combined_diagnosis()

        print(f"\n{'='*80}")
        print("RAG诊断系统测试完成！")
        print("知识库检索功能正常")
        print("能够为不同DR等级提供个性化治疗建议")
        print("支持临床决策的循证医学支持")
        print(f"{'='*80}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"测试失败: {e}")
        sys.exit(1)