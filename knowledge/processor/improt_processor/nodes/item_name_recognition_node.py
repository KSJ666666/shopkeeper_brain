from typing import List

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from knowledge.processor.improt_processor.base import BaseNode
from knowledge.processor.improt_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients


class ItemNameRecognitionNode(BaseNode):
    """"
    商品名识别节点
    """
    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        主要职责：
        1. 负责利用LLM提取商品的具体型号（名）
        2. 嵌入商品具体型号（名）
        3. 存储到Milvus中（MySQL：模糊查询的时候不会考虑语义）
        Args:
            state:

        Returns:
            更新后的状态字典
        """
        # 1.参数校验
        file_title, chunks, item_name_chunk_k ,item_name_chunk_size = self._validate_state(state)

        item_name_context = self._prepare_llm_context(chunks,item_name_chunk_k,item_name_chunk_size)

        item_name = self._recognition_item_name(item_name_context)


    def _validate_state(self, state: ImportGraphState,k:int) -> Tuple[str,List,int,int]:
        """
        校验状态必要的参数
        :param state: 图状态字典
        :return: 商品名识别节点必要的参数
         """
        file_title = state.get("file_title")

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", message="file_title is required")

        chunks = state.get("chunks")

        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", message="chunks is required and must be a list")

        itme_name_chunk_k = self.config.item_name_chunk_k

        itme_name_chunk_size = self.config.item_name_chunk_size

        if not itme_name_chunk_k or itme_name_chunk_k <= 0:
            raise ValidationError(message="商品名识别的辅助切片数不合法")

        if not itme_name_chunk_size or itme_name_chunk_size <= 0:
            raise ValidationError(message="商品名识别的辅助切片大小不合法")

        return file_title, chunks, itme_name_chunk_k ,itme_name_chunk_size

    def _prepare_llm_context(self, chunks: List, item_name_chunk_k: int, item_name_chunk_size: int) -> str:
        """
        准备商品名识别的上下文
        Args:
            chunks: 该文档的所有切块
            item_name_chunk_k: 准备使用的块数
            item_name_chunk_size: 每个块最大字符长度
        Returns:
            上下文信息
        """
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content", "")

            if len(content) > item_name_chunk_size:
                content = content[:item_name_chunk_size]

            splice_context = f"【切片】- {index}- {content}"
            final_context.append(splice_context)

        return "\n".join(final_context)


    def _recognition_item_name(self, item_name_context: str) -> str:
        """
        利用LLM提取商品的具体型号（名）
        Args:
            item_name_context: 商品名识别的上下文
        Returns:
            商品的具体型号（名）
        """
        #1.创建llm客户端
        try:
            llm_client:ChatOpenAI = AIClients.get_openai_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败,降级使用文件标题{file_title}作为商品名 {str(e)}")
            return file_title

        #2.构建提示词
        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title = file_title, context = item_name_context)

        #3.调用llm
        try:
            # 3. 调用 返回AIMessage对象
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 4. 获取AI回复的具体内容
            llm_result = llm_response.content.strip('')
            if not llm_result or llm_result == 'UNKNOWN':
                self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名兜底")
                return file_title

            self.logger.info(f"LLM为文档：{file_title} 提取的商品名：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名: {str(e)}")
            # 降级使用文件标题作为商品名
            return file_title







