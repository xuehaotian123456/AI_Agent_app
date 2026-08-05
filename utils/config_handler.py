import yaml
from utils.path_tool import get_abs_path

def load_rag_config(config_path: str = get_abs_path("config/rag.yml"),encoding: str = "utf-8"):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader) #全量加载

def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"),encoding: str = "utf-8"):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"),encoding: str = "utf-8"):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(config_path: str = get_abs_path("config/agent.yml"),encoding: str = "utf-8"):
    with open(config_path, encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_multi_agent_config(config_path: str = get_abs_path("config/multi_agent.yml"), encoding: str = "utf-8"):
    """加载 Multi-Agent 配置，文件不存在时返回默认配置"""
    try:
        with open(config_path, encoding=encoding) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        return {
            "planner": {"max_subtasks": 3},
            "retriever": {"top_k_retrieve": 10, "top_k_rerank": 3, "alpha": 0.5, "kg_enabled": True, "kg_max_entities": 3},
            "reflection": {"max_retries": 2, "confidence_threshold": 0.6},
            "summarizer": {"max_citations": 5},
            "flywheel": {"enabled": True, "redis_key_prefix": "flywheel:", "jsonl_path": "logs/flywheel_failures.jsonl", "adapt_threshold": 3},
        }

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
multi_agent_conf = load_multi_agent_config()

if __name__ == '__main__':
    print(rag_conf["chat_model_name"])