"""
RAG 输出的结构化模型定义
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class Citation(BaseModel):
    """单个引用来源"""
    content: str = Field(description="引用的原文片段")
    source: str = Field(description="资料来源文件路径或URL")
    relevance_score: Optional[float] = Field(
        default=None,
        description="相关性分数(0-1)，由重排序模型提供",
        ge=0.0,
        le=1.0
    )


class AnswerWithCitations(BaseModel):
    """带引用的回答结构"""
    answer: str = Field(
        description="基于参考资料生成的回答内容，必须简洁准确"
    )
    citations: List[Citation] = Field(
        description="回答中引用的所有资料来源列表",
        min_items=0,
        max_items=5
    )
    confidence: Optional[float] = Field(
        default=None,
        description="回答的可信度(0-1)，基于参考资料的相关性和完整性评估",
        ge=0.0,
        le=1.0
    )

    @validator('citations')
    def validate_citations(cls, v):
        """验证引用列表：确保每个引用都有内容和来源"""
        for citation in v:
            if not citation.content.strip():
                raise ValueError("引用内容不能为空")
            if not citation.source.strip():
                raise ValueError("引用来源不能为空")
        return v

    @validator('answer')
    def validate_answer(cls, v):
        """验证回答不为空"""
        if not v.strip():
            raise ValueError("回答内容不能为空")
        return v


class RAGResponse(BaseModel):
    """RAG 服务的完整响应结构（包含元数据）"""
    query: str = Field(description="用户原始查询")
    result: AnswerWithCitations = Field(description="结构化回答结果")
    retrieval_metadata: Optional[dict] = Field(
        default=None,
        description="检索元数据（如使用的检索方式、耗时等）"
    )
