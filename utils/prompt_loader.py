from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_prompt_from_file(prompt_path: str) -> str:
    """
    通用的提示词文件加载函数

    Args:
        prompt_path: 提示词文件的绝对路径或相对路径

    Returns:
        提示词文本内容
    """
    try:
        # 如果传入的是相对路径，转换为绝对路径
        if not prompt_path.startswith(('C:', 'D:', '/')):
            prompt_path = get_abs_path(prompt_path)

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.debug(f"[load_prompt_from_file] 成功加载提示词文件: {prompt_path}")
            return content
    except FileNotFoundError:
        logger.error(f"[load_prompt_from_file] 提示词文件不存在: {prompt_path}")
        raise
    except Exception as e:
        logger.error(f"[load_prompt_from_file] 读取提示词文件失败: {e}")
        raise

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt] 配置文件错误：{e}  yaml配置中没有系统提示语路径")
        raise e
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_system_prompt] 读取系统提示语失败：{e}")
        raise e

def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] 配置文件错误：{e}  yaml配置中没有rag提示语路径")
        raise e
    try:
        with open(rag_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] 读取rag提示语失败：{e}")
        raise e

def load_rag_structured_prompts():
    """加载结构化输出的 RAG prompt"""
    try:
        # 优先使用专用的结构化 prompt 文件
        structured_prompt_path = get_abs_path("prompts/rag_summarize_structured.txt")
        with open(structured_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # 如果文件不存在，降级使用传统 prompt
        logger.warning("[load_rag_structured_prompts] 结构化 prompt 文件不存在，使用传统 prompt")
        return load_rag_prompts()
    except Exception as e:
        logger.error(f"[load_rag_structured_prompts] 读取结构化 rag 提示语失败：{e}")
        raise e

def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] 配置文件错误：{e}  yaml配置中没有report_prompt_path提示语路径")
        raise e
    try:
        with open(report_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] 读取report_prompt_path提示语失败：{e}")
        raise e

def load_planner_prompt():
    """加载 Planner Agent 提示词"""
    try:
        path = get_abs_path("prompts/planner.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("[load_planner_prompt] 提示词文件不存在")
        return load_prompt_from_file(get_abs_path("prompts/planner.txt"))

def load_reflector_prompt():
    """加载 Reflector Agent 提示词"""
    try:
        path = get_abs_path("prompts/reflector.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("[load_reflector_prompt] 提示词文件不存在")
        return load_prompt_from_file(get_abs_path("prompts/reflector.txt"))

def load_summarizer_prompt():
    """加载 Summarizer Agent 提示词"""
    try:
        path = get_abs_path("prompts/summarizer.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("[load_summarizer_prompt] 提示词文件不存在")
        return load_prompt_from_file(get_abs_path("prompts/summarizer.txt"))


if __name__ == '__main__':
    print(load_system_prompts())
    print(load_rag_prompts())
    print(load_report_prompts())

