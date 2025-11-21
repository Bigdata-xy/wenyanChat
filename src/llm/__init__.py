from conf.model import LLM_BASE_URL, LLM_API_KEY

from src.llm.llm_client import OpenAICompatibleClient

llm_client = OpenAICompatibleClient(LLM_BASE_URL, LLM_API_KEY)