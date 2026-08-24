from typing import List, Dict, Any

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.improt_processor.base import BaseNode, T
from knowledge.processor.improt_processor.exceptions import StateFieldError, EmbeddingError, ValidationError, ConfigurationError
from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        对文档进行嵌入
        """
        # 1. 校验state的chunks
        self.log_step("step1", "校验chunks的数据结构")
        validated_chunks = self._validate_state(state)

        # 2. 获取嵌入模型
        self.log_step("step2", "获取BGE-M3嵌入模型客户端")

        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败,原因:{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败,原因:{str(e)}", node_name=self.name)

        #3. 对chunks进行分批次嵌入
        self.log_step("step3", "对chunks进行分批次嵌入")

        #获取单批次的chunk数量
        batch_size = self.config.embedding_batch_size
        if batch_size <= 0:
            raise ConfigurationError(
                message=f"embedding_batch_size 必须为正整数，当前值:{batch_size}",
                node_name=self.name)

        # 3.2 获取chunks的总数
        total = len(validated_chunks)

        final_chunks = []

        for chunk in range(0, total, batch_size):
            batch_chunks = validated_chunks[chunk:chunk + batch_size]

            #获取单批次最后一个chunk的索引
            batch_end = chunk + len(batch_chunks)
            self.logger.info(f"嵌入批次 [{chunk + 1}-{batch_end}] / {total}")

            # 3.3 对单批次的chunk进行嵌入
            current_chunks = self._embed_chunks(batch_chunks, embed_model)

            final_chunks.extend(current_chunks)

        # 4. 更新state的chunks
        state['chunks'] = final_chunks
        # 5. 返回
        return state



    def _validate_state(self, state: ImportGraphState) -> List[Dict[str,Any]]:
        """
        校验state中的chunks数据结构
        :param state:
        :return:
        """
        chunks = state.get("chunks", [])

        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 校验单个chunk
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"[chunk_{index + 1}] 类型和期望的类型不匹配，实际的类型{type(chunk).__name__}",
                    node_name=self.name)

        # 4. 返回chunks
        return chunks

    def _embed_chunks(self, batch_chunks: List[Dict[str,Any]], embed_model: BGEM3EmbeddingFunction) -> List[Dict[str,Any]]:
        """
        对单批次的chunk进行嵌入
        :param batch_chunks:
        :param embed_model:
        :return:
        """
        # 1. 获取要嵌入的内容
        embedding_documents = [f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}" for chunk in batch_chunks]

        #2.对内容进行嵌入
        try:
            embed_vector = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入失败,原因:{str(e)}", node_name=self.name)

        if not embed_vector:
            raise EmbeddingError(message="嵌入结果不存在")

        sparse_csr = embed_vector.get('sparse')
        # 3. 合并嵌入向量与chunk
        for i, chunk in enumerate(batch_chunks):
            chunk['dense_vector'] = embed_vector.get('dense')[i].tolist()
            chunk['sparse_vector'] = self._extract_sparse_vector(sparse_csr, i)

        # 4. 返回嵌入后的chunks
        return batch_chunks

    def _extract_sparse_vector(self, sparse_csr, index)  :
        """
        从稀疏矩阵中提取当前chunk对象的稀疏向量
        Args:
            sparse_csr:
            index:

        Returns:

        """
        # 3.1 从行索引中获取当前chunk的起始索引和结束索引
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]
        # 3.2 获取token_id
        token_id = sparse_csr.indices[start_index:end_index].tolist()
        # 3.3 获取weight
        weight = sparse_csr.data[start_index:end_index].tolist()

        # 3.4 返回单个chunk的稀疏向量值
        return dict(zip(token_id, weight))







