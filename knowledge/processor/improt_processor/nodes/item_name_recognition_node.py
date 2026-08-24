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


          #2. 准备LLM上下文
        item_name_context = self._prepare_llm_context(chunks,item_name_chunk_k,item_name_chunk_size)
             #3. 调用LLM提取商品的具体型号（名）
        item_name = self._recognition_item_name(item_name_context)

        # 4. 向量化(嵌入模型：1.OpenAIEmbedding(OpenAI) 2.文本嵌入模型（text-embedding-v(x))（灵积服务平台：dashscope） 3.bge(bge-m3))：混合向量[稠密：相似性匹配、稀疏：精确匹配]
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)


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

    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """
        商品名嵌入
        Args:
            item_name: 商品名

        Returns:
          稠密向量,稀疏向量
        """

        # 1. 获取到嵌入模型
        try:
            bge_m3_client: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE_M3嵌入模型客户端创建失败: {str(e)}")
            return None, None
        try:
            # 2. 计算稠密和稀疏向量
            vector_result = bge_m3_client.encode_documents(documents=[item_name])

            # 3. 解析稠密向量和稀疏向量
            # 3.1 获取稠密向量
            dense_vector = vector_result.get('dense')[0].tolist()
            # 3.2 获取稀疏向量矩阵csr(解开)
            sparse_csr = vector_result.get('sparse')
            # 3.1 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 3.2 获取token_id
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            # 3.3 获取weight
            weight = sparse_csr.data[start_index:end_index].tolist()
            # 3.4 构建{"token_id":weight}字典结构

            sparse_vector = dict(zip(token_id, weight))
            self.logger.info(f"计算出来的稠密向量的维度: {len(dense_vector)}")
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE-M3嵌入模型计算{item_name}向量失败 原因：{str(e)}")
            return None, None

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_name: str) -> None:
        """
        将LLM识别到的商品名保存到Milvus数据库中
        row行记录{"dense_vector":"值","sparse_vector":"值","file_title":"文档","item_name":"商品名"}
        Args:
            dense_vector: 商品名稠密向量
            sparse_vector: 商品名稀疏向量
            file_title: 文档名
            item_name:   商品名

        Returns:
        """

        # 1. 判断稠密向量和稀疏向量是否都存在
        if not dense_vector or not sparse_vector:
            return


        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败,原因：{str(e)}")
            return

        # 3. Milvus三大核心概念（集合：Collection[1.集合名 2.字段约束：schema 3.索引]）
        # 3.1 集合名：表名类似于归纳数据的容器，逻辑概念
        # 3.2 约束：类似于MySQL字段的长度、字段的类型..
        # 3.3 索引：类似于MySQL中的索引【索引类型比较多 B+树 Hash】。Milvus索引类型也有很多（专门针对于稠密向量的索引类型 针对于稀疏向量的索引类型 标量字段类型、主键类型）
        # 索引：本质就是算法（图、树、hash..）目的：提高检索【查询】效率。milvus中不管稠密向量索引还是稀疏向量的索引都是为了能够快速找到和问题相似的向量。
        # 使用Milvus的流程：①：创建集合（约束、索引）②：插入数据  ③：查询/检索

        # 4. 获取集合名字
        item_name_collection_name = self.config.item_name_collection
        try:
            # 5. 创建Milvus集合(幂等性校验)
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)

            # 6. 构建数据行
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            # 7. 插入数据
            inserted_result = milvus_client.insert(collection_name=item_name_collection_name, data=[item_name_data_row])
        except Exception as e:
            self.logger.error(f"商品名{item_name}插入失败 {str(e)}")

        self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")








