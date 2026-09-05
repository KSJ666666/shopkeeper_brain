import os
import shutil
import uuid
import logging
from datetime import datetime

import logger
from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.improt_processor.exceptions import FileProcessingError
from knowledge.processor.improt_processor.main_graph import import_app
from knowledge.utils.client.storage_clients import StorageClients


class UploadService:
    """文件上传服务"""


    def process_upload_file(self, file: UploadFile):
        """
        处理上传文件
        1.讲上传的文件存入本地临时目录
        2.将上传的文件存入远程minio中
        3.返回上传文件的路径
        :param file:
        :return:
        """
        #1.构建任务id
        task_id = str(uuid.uuid4().hex[:8])
        #2.构建日期目录并且拼接到临时目录
        base_file_dir = self.get_base_dir()
        #3.构建file_dir  如：knowledge/temp_data/20251020/task_id
        file_dir = os.path.join(base_file_dir, task_id)
        #4.构建import_file_path  如：knowledge/temp_data/20251020/task_id/合同.pdf  并将文件存入本地临时目录
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        #5.文件上传至minio中
        self.save_upload_file_to_minio(import_file_path, file.filename)

        # 6. 返回图谱的信息
        return task_id, import_file_path, file_dir

    def run_import_graph(self, task_id, import_file_path, file_dir):
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir

        }

        # stream:迭代整个graph图状态可以得到每一个节点的事件(节点的名字以及节点操作完state之后的新状态)

        for event in import_app.stream(graph_state):
            for key, value in event.items():
                logger.info(f"当前正在执行的节点：{key}")



    def get_base_dir(self):
        """获取本地文件存储基础目录
        构建成knowledge/temp_data/日期目录
        """
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str):
        """
        将上传的文件存入本地临时目录
        :param file:
        :param file_dir:
        :return:
        """
        #1.创建目录
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建导入文件的路径
        import_file_path = os.path.join(file_dir, file.filename)
        # 3. 写入
        try:
            with  open(import_file_path, "wb") as f:
                # shutil.copyfileobj() 不同的操作系统以及不同python版本都可以分批次的写入（windows版本以及3.7以上的sdk版本:1m）
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败 原因:{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败 原因:{str(e)}")

        return import_file_path

    def save_upload_file_to_minio(self, import_file_path: str, filename:str):
        """
        将上传的文件存入minio中
        :param import_file_path:
        :param filename:
        :return:
        """
        #1.获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.info(f"{filename.filename}上传至minio失败 原因:{str(e)}")
            return
        #2.获取minio的相关信息
        minio_bucket = os.getenv("MINIO_BUCKET_NAME")

        minio_object = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        #3.上传文件
        try:
            minio_client.fput_object(minio_bucket,minio_object,import_file_path)
        except Exception as e:
            logger.info(f"{filename}上传至minio失败 原因:{str(e)}")






