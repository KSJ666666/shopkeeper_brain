from pathlib import Path

from knowledge.processor.improt_processor.base import BaseNode, T
from knowledge.processor.improt_processor.exceptions import StateFieldError
from knowledge.processor.improt_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name = "entry_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
         根据上传文件的后缀 修改state中 is_md_read_enabled  is_pdf_read_enabled
        :param state:
        :return:
        """
        #1. 从state中获取上传文件的路径
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")
        #2.判断是否为空
        if not import_file_path :
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)
        if not file_dir :
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str)
        #3.path标准化
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        #4.判断文件是否存在
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)

        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str)

        #5.判断文件后缀
        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state['pdf_path'] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state['md_path'] = str(import_file_path_obj)
        else:
            self.logger.error(f"不支持的文件后缀: {import_file_path_obj.suffix}")
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)

            # 6. 获取上传文件的标题，更新到state中
        state['file_title'] = import_file_path_obj.stem

        return state
















