import threading
from typing import Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()


class AIClients(BaseClientManager):
    """AI 模型类客户端：OpenAI(VLM)"""
    # ── OpenAI / VLM ──

    _openai_vlm_client: Optional[OpenAI] = None
    _openai_vlm_lock = threading.Lock()

    # ── OpenAI / LLM ──
    _openai_llm_client: Optional[ChatOpenAI] = None
    _openai_llm_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()


    @classmethod
    def get_openai_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_vlm_client", cls._openai_vlm_lock, cls._create_openai_vlm_client)

    @classmethod
    def _create_openai_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    @classmethod
    def get_openai_llm_client(cls,response_format:bool = True) -> ChatOpenAI:
        return cls._get_or_create("_openai_llm_client", cls._openai_llm_lock, lambda : cls._create_openai_llm_client(response_format))

    @classmethod
    def _create_openai_llm_client(cls,response_format:bool = True) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            modle = cls._require_env("LLM_DEFAULT_MODEL")

            konw_kwargs = {
            }
            if response_format:
                konw_kwargs["response_format"] = {
                    "type": "json_object",
                }

            client = ChatOpenAI(
                temperature=0,
                model=modle,
                api_key=api_key,
                base_url=base_url,
                model_kwargs=konw_kwargs
            )
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Llm 客户端创建失败: {e}")
            raise ConnectionError(f"Llm 连接失败: {e}") from e

        # ── BGE-M3嵌入模型客户端 ──
    @classmethod
    def get_bge_m3_client(cls):
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls):
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name = cls._require_env('BGE_M3_PATH')
            device = cls._require_env('BGE_DEVICE')
            fp16_str = cls._require_env('BGE_FP16')

            fp16 = fp16_str.lower() in ("true", "1")
            # 2. 创建
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e


if __name__ == '__main__':
    llm_client:ChatOpenAI = AIClients.get_openai_llm_client()
    response = llm_client.invoke("给我讲一句笑话，要求输出json格式")
    import json
    data = json.loads(response.content)
    print(data)
