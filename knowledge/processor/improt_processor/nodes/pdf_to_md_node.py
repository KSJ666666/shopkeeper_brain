import subprocess, time, json, sys

from pathlib import Path
from typing import Tuple

from knowledge.processor.improt_processor.base import BaseNode, T, setup_logging
from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.processor.improt_processor.exceptions import StateFieldError,FileProcessingError



class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:


        """
        核心逻辑：获取pdf文件上传路径，利用解析工具将pdf装化成md
        :param state:
        :return:
        """
        #1.验证状态，返回pdf文件上传路径和处理目录
        import_file_path_obj, file_dir_obj = self._validate_state(state)

        #2.调用解析工具，将pdf文件解析为md文件
        process_code = self._execute_mineru_parse(import_file_path_obj,file_dir_obj)
        if process_code != 0:
            raise FileProcessingError(f"解析工具执行失败，退出码：{process_code}")

        #3.获取解析得到的md文件路径
        md_ath = self._get_md_fire_path(import_file_path_obj,file_dir_obj)

        #4.将md文件路径添加到状态中并更新状态
        state["md_path"] = md_ath
        return  state

    def _validate_state(self,state:ImportGraphState)->Tuple[Path,Path]:
        """
        验证状态，返回pdf文件上传路径和处理目录
        :param state:
        :return:
        """
        #取出文件上传路径
        import_file_path = state.get("import_file_path",'')
        if not import_file_path:
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str,message="pdf文件上传路径缺失或无效")
        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str,message="pdf文件 不存在")

        #取出处理目录
        file_dir = state.get("file_dir",'')
        if not file_dir:
            file_dir = import_file_path_obj.parent
        file_dir_obj = Path(file_dir)

        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name,field_name="file_dir",expected_type=str,message="输出目录 不存在")

        self.logger.info(f"解析的文件路径{import_file_path_obj}")
        self.logger.info(f"输出的文件目录{file_dir_obj}")

        return import_file_path_obj,file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj, file_dir_obj):
        """
        调用解析工具，将pdf文件解析为md文件
        :param import_file_path_obj:
        :param file_dir_obj:
        :return:
        """
        # 调用解析工具，将pdf文件解析为md文件
        #1.定义cmd
        mineru_exe = str(Path(sys.executable).parent / "Scripts" / "mineru.exe")
        cmd = [
            mineru_exe,
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
            "--source",
            "local"
        ]
        # 2. 利用子进程执行cmd命令(子进程解析的日志【正常日志和错误日志都要】)
        # 子进程（执行命令产生的日志[正常、错误日志]）-----管子----外部线程（_execute_mineru_parse）
        start_time = time.time()
        proc = subprocess.Popen(args=cmd,
                                stdout=subprocess.PIPE,  # 接收正确日志
                                stderr=subprocess.STDOUT,  # 接收错误日志
                                text=True,  # 输出二进制字节流，输出字符串
                                errors="replace",  # 特殊的字符码替换成?、菱形
                                encoding="utf-8",  # utf-8进行解码
                                bufsize=1  # 实时输出。按行输出遇到\n换行符 就将日志产生出来
                                )
        #逐行打印日志
        for line in proc.stdout:
            self.logger.info(f"MinerU解析产生的日志：{line}")
        # 3.等待子进程执行完成
        process_result = proc.wait()
        end_time = time.time()
        # 4.解析子进程执行结果
        if process_result == 0:
            self.logger.info(f"解析工具执行成功，解析耗时：{end_time-start_time}")
        else:
            self.logger.error(f"解析工具执行失败，退出码：{process_result}")

        return process_result

    def _get_md_fire_path(self, import_file_path_obj, file_dir_obj):
        """
        获取解析后md的路径
        :param import_file_path_obj:
        :param file_dir_obj:
        :return:
        md_path= D:\develop\develop\workspace\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md

        Path:吉祥三包：name:全名[文件名字.后缀]   stem【文件名字，没有后缀】    suffix【文件的后缀】

        """
        #1.获取文件名
        file_name = import_file_path_obj.stem

        #2.获取md文件路径
        return str(file_dir_obj / file_name / "hybrid_auto" / f"{file_name}.md")


###########################################
# 测试
###########################################
if __name__ == '__main__':
    setup_logging()
    # 1. 构建节点实例
    pdf_to_md_node = PdfToMdNode()
    # 2. 构建该节点状态
    init_state = {
        "import_file_path": r"D:\agent资料\sgg_zuixin\尚硅谷 - 大模型【7月结课】\17_尚硅谷大模型项目之掌柜智库\2.资料\pdf文档\doc\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf",
        "file_dir": r"D:\agent资料\sgg_zuixin\尚硅谷 - 大模型【7月结课】\17_尚硅谷大模型项目之掌柜智库\2.资料\pdf文档\doc"
    }
    # 3. 直接调用process
    result = pdf_to_md_node.process(init_state)
    # 4. 序列化（将对象转成字符串） 反序列化（将字符串转成对象）
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)













