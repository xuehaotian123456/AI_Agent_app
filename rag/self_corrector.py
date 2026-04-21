"""
自我纠错（Corrective RAG）模块
在生成答案后进行质量检查，必要时重新检索和生成
"""
from typing import Optional, Dict, Any
from langchain_core.prompts import PromptTemplate
from model.factory import robust_llm_caller
from utils.logger_handler import logger
import re
import time


class SelfCorrector:
    """自我纠错器：检查答案质量并在必要时重新生成"""

    def __init__(self, max_retries: int = 2, confidence_threshold: float = 0.6):
        """
        初始化自我纠错器

        Args:
            max_retries: 最大重试次数（避免无限循环）
            confidence_threshold: 置信度阈值，低于此值触发重试
        """
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

        # 加载质量检查提示词
        self.check_prompt_template = self._load_check_prompt()

        # 查询改写提示词
        self.rewrite_prompt_template = self._load_rewrite_prompt()

        logger.info(f"[SelfCorrector] 初始化完成 | 最大重试: {max_retries}, 置信度阈值: {confidence_threshold}")

    def _load_check_prompt(self) -> PromptTemplate:
        """加载质量检查提示词"""
        try:
            from utils.prompt_loader import load_prompt_from_file
            from utils.path_tool import get_abs_path

            prompt_path = get_abs_path("prompts/self_check.txt")
            template = load_prompt_from_file(prompt_path)
            return PromptTemplate.from_template(template)
        except Exception as e:
            logger.warning(f"[SelfCorrector] 加载提示词文件失败，使用默认模板: {e}")
            # 降级：使用内置模板
            default_template = """你是答案质量检查专家。

用户问题：{query}
参考资料：{context}
生成答案：{answer}

请判断答案质量，输出格式：
【判定结果】通过/需要重试
【原因说明】<简要说明>

评估标准：准确性、完整性、引用充分性、相关性"""
            return PromptTemplate.from_template(default_template)

    def _load_rewrite_prompt(self) -> PromptTemplate:
        """加载查询改写提示词"""
        template = """你是一个查询优化专家。原始查询未能获得满意的答案，请根据以下信息改写查询，使其更具体、更容易检索到相关信息。

原始查询：{original_query}
已检索资料：{context}
初次答案：{initial_answer}
质量检查结果：{check_reason}

请生成一个改进后的查询，要求：
1. 简洁明了（不超过30个字）
2. 包含核心关键词和术语
3. 避免冗长的描述和文档名称
4. 聚焦用户真正想了解的问题
5. 使用自然语言，适合向量相似度匹配

只输出改写后的查询文本，不要添加任何前缀或解释。


改进后的查询："""
        return PromptTemplate.from_template(template)

    def check_answer_quality(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """
        检查答案质量

        Args:
            query: 用户查询
            context: 检索到的上下文
            answer: 生成的答案

        Returns:
            质量检查结果字典
        """
        start_time = time.time()

        try:
            # 格式化检查提示词
            check_prompt = self.check_prompt_template.format(
                query=query,
                context=context[:2000],  # 限制上下文长度，避免超出 token 限制
                answer=answer
            )

            # 调用 LLM 进行质量检查
            logger.debug("[SelfCorrector] 开始质量检查...")
            response = robust_llm_caller(check_prompt)
            check_result = response.content if hasattr(response, 'content') else str(response)

            # 解析检查结果
            parsed = self._parse_check_result(check_result)
            elapsed = time.time() - start_time

            parsed['check_time'] = round(elapsed, 3)
            parsed['raw_output'] = check_result

            logger.info(
                f"[SelfCorrector] 质量检查完成 | "
                f"判定: {parsed['verdict']} | "
                f"耗时: {elapsed:.2f}s | "
                f"原因: {parsed.get('reason', 'N/A')[:50]}"
            )

            return parsed

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[SelfCorrector] 质量检查失败: {e}", exc_info=True)
            # 检查失败时，默认通过（避免阻塞正常流程）
            return {
                'verdict': '通过',
                'reason': f'质量检查异常，默认通过: {str(e)[:100]}',
                'check_time': round(elapsed, 3),
                'raw_output': ''
            }

    def _parse_check_result(self, check_output: str) -> Dict[str, str]:
        """
        解析质量检查结果

        Args:
            check_output: LLM 输出的检查结果文本

        Returns:
            解析后的字典
        """
        result = {
            'verdict': '通过',  # 默认通过
            'reason': '',
            'suggestion': ''
        }

        try:
            # 提取判定结果
            verdict_match = re.search(r'【判定结果】\s*(通过|需要重试)', check_output)
            if verdict_match:
                result['verdict'] = verdict_match.group(1)
            else:
                # 兼容其他格式
                if '需要重试' in check_output or '不通过' in check_output:
                    result['verdict'] = '需要重试'
                elif '通过' in check_output:
                    result['verdict'] = '通过'

            # 提取原因说明
            reason_match = re.search(r'【原因说明】\s*(.+?)(?:\n|$)', check_output)
            if reason_match:
                result['reason'] = reason_match.group(1).strip()

            # 提取改进建议
            suggestion_match = re.search(r'【改进建议】\s*(.+?)(?:\n|$)', check_output)
            if suggestion_match:
                result['suggestion'] = suggestion_match.group(1).strip()

        except Exception as e:
            logger.warning(f"[SelfCorrector] 解析检查结果失败: {e}，使用默认值")

        return result

    def rewrite_query(self, original_query: str, context: str,
                      initial_answer: str, check_reason: str) -> str:
        """
        改写查询以获得更好的检索结果

        Args:
            original_query: 原始查询
            context: 已检索的上下文
            initial_answer: 初次生成的答案
            check_reason: 质量检查的原因说明

        Returns:
            改写后的查询
        """
        try:
            rewrite_prompt = self.rewrite_prompt_template.format(
                original_query=original_query,
                context=context[:1000],
                initial_answer=initial_answer[:500],
                check_reason=check_reason[:200]
            )

            logger.debug("[SelfCorrector] 开始改写查询...")
            response = robust_llm_caller(rewrite_prompt)
            new_query = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            # 清理可能的多余内容，提取实际查询
            lines = [line.strip() for line in new_query.split('\n') if line.strip()]

            # 过滤掉前缀标记，提取真正的查询内容
            cleaned_query = ""
            for line in lines:
                # 跳过空行和前缀标记
                if not line or line.startswith('改进后的查询：') or line.startswith('改进后的查询:'):
                    # 如果这一行包含冒号后面的内容，提取出来
                    if '：' in line:
                        cleaned_query = line.split('：', 1)[1].strip()
                    elif ':' in line:
                        cleaned_query = line.split(':', 1)[1].strip()
                    if cleaned_query:
                        break
                else:
                    # 直接取第一行非空内容
                    cleaned_query = line
                    break

            # 如果没有提取到内容，使用原始查询
            if not cleaned_query or len(cleaned_query) < 5:
                logger.warning(f"[SelfCorrector] 查询改写结果为空或过短，使用原始查询")
                return original_query

            # 限制长度，避免过长的查询影响检索效果
            max_query_length = 50
            if len(cleaned_query) > max_query_length:
                logger.warning(
                    f"[SelfCorrector] 查询改写结果过长 ({len(cleaned_query)}字)，截断至{max_query_length}字"
                )
                cleaned_query = cleaned_query[:max_query_length]

                # 尝试在合适的边界截断（句号、逗号、空格）
                for sep in ['。', '，', ',', ' ', '、']:
                    last_sep = cleaned_query.rfind(sep)
                    if last_sep > max_query_length * 0.6:  # 至少在60%位置有分隔符
                        cleaned_query = cleaned_query[:last_sep]
                        break

            logger.info(
                f"[SelfCorrector] 查询改写完成 | 原: '{original_query[:50]}...' | 新: '{cleaned_query[:50]}...'")

            return cleaned_query

        except Exception as e:
            logger.error(f"[SelfCorrector] 查询改写失败: {e}，返回原始查询")
            return original_query

    def should_retry(self, quality_check: Dict[str, Any], retry_count: int) -> bool:
        """
        判断是否应该重试

        Args:
            quality_check: 质量检查结果
            retry_count: 当前重试次数

        Returns:
            是否应该重试
        """
        # 检查是否达到最大重试次数
        if retry_count >= self.max_retries:
            logger.warning(f"[SelfCorrector] 已达到最大重试次数 ({self.max_retries})，停止重试")
            return False

        # 检查判定结果
        if quality_check['verdict'] == '需要重试':
            return True

        return False


class CorrectiveRAGService:
    """
    带自我纠错的 RAG 服务
    包装原有的 RagSummarizeService，在其基础上添加纠错能力
    """

    def __init__(self, rag_service, enable_correction: bool = True,
                 max_retries: int = 2, confidence_threshold: float = 0.6):
        """
        初始化带纠错的 RAG 服务

        Args:
            rag_service: 原始的 RagSummarizeService 实例
            enable_correction: 是否启用自我纠错
            max_retries: 最大重试次数
            confidence_threshold: 置信度阈值
        """
        self.rag_service = rag_service
        self.enable_correction = enable_correction
        self.corrector = SelfCorrector(
            max_retries=max_retries,
            confidence_threshold=confidence_threshold
        )

        logger.info(
            f"[CorrectiveRAG] 初始化完成 | "
            f"纠错启用: {enable_correction} | "
            f"最大重试: {max_retries}"
        )

    def _fallback_retrieval_metadata(self, query: str) -> Dict[str, Any]:
        """在上游未返回检索元数据时，提供兜底元数据。"""
        try:
            docs = self.rag_service.retriever_docs(query)
            retrieval_method = "hybrid" if (
                getattr(self.rag_service.vector_store, "hybrid_enabled", False) and
                hasattr(self.rag_service.retriever, "hybrid_search")
            ) else "vector"
            return {
                "method": retrieval_method,
                "docs_count": len(docs),
                "elapsed_time": None
            }
        except Exception as e:
            logger.warning(f"[CorrectiveRAG] 兜底检索元数据失败: {e}")
            return {
                "method": "unknown",
                "docs_count": 0,
                "elapsed_time": None
            }

    def rag_summarize_with_correction(
        self,
        query: str,
        thread_id: str = "default",
        enable_correction: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        执行带自我纠错的 RAG 总结

        Args:
            query: 用户查询
            thread_id: 会话ID

        Returns:
            包含答案和纠错信息的字典
        """
        effective_enable_correction = self.enable_correction if enable_correction is None else enable_correction

        if not effective_enable_correction:
            # 未启用纠错，直接调用原服务
            llm_usage = {}
            if hasattr(self.rag_service, 'rag_summarize_structured'):
                base_result = self.rag_service.rag_summarize_structured(query, thread_id)
                answer = base_result.get("answer", "")
                retrieval_metadata = base_result.get("retrieval_metadata") or self._fallback_retrieval_metadata(query)
                structured_result = base_result.get("structured_result")
                model_used = base_result.get("generation_model_used", "unknown")
                llm_usage = base_result.get("llm_usage", {})
            else:
                answer = self.rag_service.rag_summarize(query, thread_id)
                retrieval_metadata = {}
                structured_result = None
                model_used = "unknown"

            return {
                'answer': answer,
                'corrected': False,
                'retry_count': 0,
                'quality_checks': [],
                'final_query': query,
                'retrieval_metadata': retrieval_metadata,
                'structured_result': structured_result,
                'model_used': model_used,
                'llm_usage': llm_usage
            }

        # 启用纠错，执行带重试的流程
        current_query = query
        retry_count = 0
        quality_checks = []
        final_answer = None
        final_context = None
        last_structured_result = None

        while retry_count <= self.corrector.max_retries:
            logger.info(
                f"[CorrectiveRAG] 第 {retry_count + 1} 轮检索生成 | "
                f"查询: '{current_query[:50]}...'"
            )

            # 执行 RAG 检索和生成
            if hasattr(self.rag_service, 'rag_summarize_structured'):
                # 使用结构化输出
                result = self.rag_service.rag_summarize_structured(current_query, thread_id)
                answer = result['answer']
                context = result.get('context', '')
                last_structured_result = result  # 保存最后一次的结构化结果
            else:
                # 使用普通输出
                answer = self.rag_service.rag_summarize(current_query, thread_id)
                # 获取上下文（需要从内部状态获取，这里简化处理）
                context = ""

            # 质量检查
            quality_check = self.corrector.check_answer_quality(
                query=current_query,
                context=context,
                answer=answer
            )
            quality_checks.append(quality_check)

            # 判断是否需要重试
            if not self.corrector.should_retry(quality_check, retry_count):
                logger.info(f"[CorrectiveRAG] 质量检查通过，返回答案")
                final_answer = answer
                final_context = context
                break

            # 需要重试：改写查询
            logger.warning(
                f"[CorrectiveRAG] 质量检查未通过，准备第 {retry_count + 2} 轮重试 | "
                f"原因: {quality_check.get('reason', 'N/A')[:100]}"
            )

            # 改写查询
            current_query = self.corrector.rewrite_query(
                original_query=current_query,
                context=context,
                initial_answer=answer,
                check_reason=quality_check.get('reason', '')
            )

            retry_count += 1

        # 构建返回结果，直接使用循环中保存的结果，避免重复调用
        result = {
            'answer': final_answer or answer,
            'corrected': retry_count > 0,
            'retry_count': retry_count,
            'quality_checks': quality_checks,
            'final_query': current_query if retry_count > 0 else query
        }

        # 添加最后一次检索的元数据（避免重复调用）
        if last_structured_result:
            result['structured_result'] = last_structured_result.get('structured_result')
            result['retrieval_metadata'] = (
                last_structured_result.get('retrieval_metadata')
                or self._fallback_retrieval_metadata(current_query)
            )
            result['model_used'] = last_structured_result.get('generation_model_used', 'unknown')
            result['llm_usage'] = last_structured_result.get('llm_usage', {})
        else:
            result['retrieval_metadata'] = self._fallback_retrieval_metadata(current_query)
            result['model_used'] = "unknown"
            result['llm_usage'] = {}

        logger.info(
            f"[CorrectiveRAG] 完成 | "
            f"重试次数: {retry_count} | "
            f"最终查询: '{result.get('final_query', query)[:50]}...'"
        )

        return result

    def rag_summarize(self, query: str, thread_id: str = "default") -> str:
        """
        兼容原有接口的简单调用方法

        Args:
            query: 用户查询
            thread_id: 会话ID

        Returns:
            答案字符串
        """
        result = self.rag_summarize_with_correction(query, thread_id)
        return result['answer']
