from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional, Sequence

from pymilvus import MilvusClient, DataType

from knowledge.processor.improt_processor.base import BaseNode, T
from knowledge.processor.improt_processor.exceptions import StateFieldError, ValidationError, MilvusError
from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)
class _MilvusSchemaBuilder():
   """
   负责构建MilVus集合的schema
   """

   @staticmethod
   def build_schema(dim: int) -> List[Dict[str, Any]]:
       """
       构建MilVus集合的schema

       Args:
           dim: 向量维度

       Returns:
           List[Dict[str, Any]]: MilVus集合的schema
       """
       return [
           {"name": "chunk_id", "type": "Int64"},
           {"name": "dense_vector", "type": "FloatVector", "dim": dim},
           {"name": "sparse_vector", "type": "FloatVector", "dim": 1000000},
       ]


class ImportMilvusNode(BaseNode):
    """
    角色：充当门面（门面模式）
    """

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 校验state
        validated_chunks, dim = self._validate_state(state)

        # 2.创建milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"创建milvus客户端失败: {e}")
            raise MilvusError(message=f"MilVus客户端创建失败,异常原因{str(e)}", node_name=self.name)

        #3.构建milvus集合
        collection_name = self.config.chunks_collection

        self.create_milvus_colletction(collection_name, dim, milvus_client)

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:

        self.log_step("validate", "参数校验")
        # 获取chunks
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError("待入库的 chunks 为空或类型无效", self.name)

        final_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                final_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not final_chunks:
            raise ValidationError("待入库的 chunks 无可用向量", self.name)

        dim = len(final_chunks[0]["dense_vector"])

        return final_chunks, dim

    def create_milvus_colletction(self, collection_name: str, dim: int, milvus_client: MilvusClient):
        #1.判断集合是否存在
        if milvus_client.has_collection(collection_name):
            self.logger.info(f"集合 {collection_name} 已存在，无需创建")
            return

        #2.创建集合
        #2.1创建schema
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)


