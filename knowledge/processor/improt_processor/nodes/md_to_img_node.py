import os
import sys

if __name__ == "__main__":
    # 独立运行本文件时把项目根加入 sys.path，保证 knowledge 包可导入
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Set, Dict, Deque

from openai import OpenAI

from knowledge.processor.improt_processor.base import BaseNode
from knowledge.processor.improt_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


@dataclass
class ImageContext:
    """
    图片上下文信息
    """
    head:str  #上文标题内容
    pre_text:str  #上文内容
    post_text:str  #下文内容

@dataclass
class ImageInfo:

    """
    图片完整信息
    """
    name:str #图片名称
    path:str #图片路径
    image_context:ImageContext #图片上下文信息

class _MdFileHandler:
    """
    处理Markdown文件的类
    """
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def validate_and_read_md(self, state: ImportGraphState)->Tuple[str,Path,Path]:
        """
        校验并读取Markdown文件，返回文件内容，文件路径，图片目录
        :param state:
        :return:
        """
        #1.获取md路径
        md_path = state.get("md_path","")
        #2.校验md路径是否为空
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=str, message="Markdown文件路径不能为空")

        #3。将md路径转换为Path对象
        md_path_obj = Path(md_path)
        #4.校验路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=str, message="Markdown文件路径不存在")

        #5.读取Markdown文件内容
        try:
            with open(md_path_obj, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except IOError as e:
            self.logger(f"MD文件:{md_path_obj.name} 打开失败")
            raise FileProcessingError(message=f"MD文件:{md_path_obj.name} 打开失败")

        #6.获取图片的目录
        img_dir = md_path_obj.parent /"images"

        #7.返回
        return md_content,md_path_obj,img_dir




class _ImageScanner:
    """
    图片扫描器类
    """
    def __init__(self, logger):
        self.logger = logger

    def scan_imgs_dir(self, img_dir_obj:Path, md_content:str, image_extensions:Set[str], img_content_length:int)->List[ImageInfo]:
        """
        扫描图片目录，返回图片完整信息列表
        :param img_dir_obj: 图片目录
        :param md_content: md文件内容
        :param image_extensions: 图片扩展名集合
        :param img_content_length: 图片上下文最大长度
        :return: 图片完整信息列表
        """

        img_info_list = []
        for img_path in img_dir_obj.iterdir():
            if not img_path.is_file():
                self.logger.error(f"{img_path}不是一个有效的文件")
                continue

                # 1.2  过滤掉不合法的图片文件
            if not img_path.suffix in image_extensions:
                self.logger.error(f"{img_path.suffix}不是允许的图片后缀格式")
                continue
              #1.3查找是图片的上下文
            ctx = self._find_context(img_path.name,md_content,img_content_length)

            if not ctx:
                self.logger.info(f"MD中未找到该图片{img_path.name}引用")
                continue

                # 1.4 封装ImageInfo对象并且放到容器中
            img_info_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                image_context=ctx
            ))

        self.logger.info(f"MD中找到{len(img_info_list)}个有效的图片引用")

        # 2. 最终返回
        return img_info_list

    def _find_context(self, img_name, md_content, img_content_length):
        """
        查找图片上下文
        :param img_name: 图片名称
        :param md_content: md文件内容
        :param img_content_length: 图片上下文最大长度
        :return: 图片上下文信息
        """
        # 1. 预编译正则规则(主要目的：从MD（很多行）中抓取到当前这个图片)
        # ![](images\xxx.png "abc")
        # 正则在大模型应用中特别多
        # . 任意字符 * 0次或者多次  \[ \] \( \) ?非贪婪模式  escape（a.png）
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        md_lines = md_content.split("\n")
        #1.遍历md文件内容，先查找图片出现的位置索引
        for md_idx,md_line in enumerate(md_lines):
            if not pattern.search(md_line):
                continue
            #2.根据图片出现的位置索引，获取上下文
            head_line,prev_index = self._find_heading_up(md_lines, md_idx)
            pre_lines = md_lines[prev_index + 1:md_idx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction="front")

            #3.根据图片出现的位置索引，获取下文
            down_result = self._find_heading_down(md_lines, md_idx)
            next_index = down_result[-1]
            post_lines = md_lines[md_idx + 1:next_index]
            post_context = self._extract_limited_context(post_lines, img_content_length, direction="back")

            return ImageContext(
                head=head_line,
                pre_text=pre_context,
                post_text=post_context
            )
            return None



    def _find_heading_up(self, md_lines : List[str], md_idx:int)->Tuple[str,int]:
        """
        查找图片上下文的标题
        :param md_lines: md文件内容列表
        :param md_idx: 图片所在行索引
        :return: 标题内容，标题索引
        """
        for i in range(md_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    def _find_heading_down(self, md_lines, md_idx):
        """
        查找图片下文的标题
        :param md_lines: md文件内容列表
        :param md_idx: 图片所在行索引
        :return: 标题内容，标题索引
        """
        for i in range(md_idx + 1,len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", len(md_lines)

    def _extract_limited_context(self, extracted_md_lines:List[str], img_content_length:int, direction:str)->str:
        current_paragraph = []
        paragraphs = []

        # 1. 遍历截取的行
        for line in extracted_md_lines:
            # 1.1 定义自然而然段落的规则
            is_blank_line = not line.strip()

            # 1.2 定义人为设计的图片段落规则
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            # 1.3 当前行是空行或者其它图片行
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            # 1.4  当前行不是空行也不是其它图片行
            current_paragraph.append(line)

        # 2. 处理最后的行
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 反转(就近原则)
        if direction == "front":
            paragraphs.reverse()
        # 3. 遍历段落列表(判断长度，已经最终选择留下哪些段落)
        total = 0
        selected = []  # 最终收集到的段落
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break
            selected.append(paragraph)
            total += len(paragraph)

        # 反转（保证收集到的顺序和原文档中顺序一致，方便VLM参考）
        if direction == "front":
            selected.reverse()

        # 4. 将最终段落列表中的段落转成一个字符串
        return "\n\n".join(selected)


class _VLMSummarizer:
    """
    LLM摘要器类
    """
    def __init__(self, logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def _summary_all(self, md_name:str, image_info_list:List[ImageInfo], vl_model:str)->Dict[str,str]:
        """
        提取图片摘要
        :param md_name: md文件名
        :param image_info_list: 图片信息列表
        :param vl_model: LLM模型
        :return: Dict[str,str]:{"img_name":"summary"}
        """
        summaries = {}
        request_timestamps: Deque[float] = deque()
        #1.获取客户端
        try:
            vlm_client = AIClients.get_openai()
        except Exception as e:
            for image_info in image_info_list:
                summaries[image_info.img_name] = "暂无摘要"
            return summaries
        #2.遍历图片信息列表，提取摘要
        for img_info in image_info_list:
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summaries[img_info.name] = self._summary_one(md_name, img_info, vlm_client, vl_model)

        return summaries

    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)

    def _summary_one(self, md_name:str, img_info:ImageInfo, vlm_client:OpenAI, vl_model:str)->str:
        """
        提取图片摘要
        :param md_name: md文件名
        :param img_info: 图片信息
        :param vlm_client: LLM客户端
        :param vl_model: LLM模型
        :return: 图片摘要
        """
        #1.构建上下文
        parts = [ p for p in (img_info.image_context.head,img_info.image_context.post_text,img_info.image_context.pre_text) if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容（二进制字节流）---文本协议认识（base64编码）--->解码（‘utf-8’）--->字符串（文本协议能传输） ---- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with  open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f"读取图片文件{img_info.path} 内容失败: {e}")
            return "暂无图片描述"
        # 4. 根据图片后缀动态设置 MIME 类型
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime = mime_map.get(Path(img_info.path).suffix.lower(), "image/jpeg")

        # 5. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{md_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"


class _ImageUploader:
    """图片上传器（预留：将图片上传到 MinIO）"""
    def __init__(self, logger):
        self.logger = logger



class MarkdownToImageNode(BaseNode):
    """
    总节点类，主要逻辑
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法
    """
    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(self.logger, self.name)
        self._img_scaner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._img_uploader = _ImageUploader(self.logger)
    name = "md_to_img_node"
    def process(self,state: ImportGraphState)->ImportGraphState:
        """
        处理方法，主要逻辑
        1. 得到四个类的实例对象
        """
        config = self.config
        self.log_step("step1","读取Markdown文件，得到文件内容，文件路径，图片目录")


        md_content,md_path_obj,img_dir_obj = self._md_file_handler.validate_and_read_md(state)

        self.log_step("step2","准备需要扫面的图片目录")

        if not img_dir_obj.exists():  #解析后，图片目录不存在 此时不需要vlm扫描图片，直接更新字段名即可
            state['md_content'] = md_content
            return state

        #准备待扫描的图片
        image_info_list : List[ImageInfo] = self._img_scaner.scan_imgs_dir(img_dir_obj,
                                                                           md_content,
                                                                           config.image_extensions
                                                                           ,config.img_content_length)
        # 3. 操作_vlm_summarizer

        self.log_step("step3", "利用VLM提取摘要")

        summaries : Dict[str,str] = self._vlm_summarizer._summary_all(md_path_obj.stem, image_info_list
                                          ,config.vl_model)

        # 4. 将摘要写回状态，供后续切片/向量化使用
        state['md_content'] = md_content
        state['image_summaries'] = summaries
        return state


if __name__ == "__main__":
    import os
    import sys
    import logging
    from dotenv import load_dotenv

    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # 1. 保证项目根可导入 + 加载环境变量（DashScope / VLM 模型）
    PROJECT_ROOT = r"D:\code\shopkeeper_brain"
    sys.path.insert(0, PROJECT_ROOT)
    env_path = os.path.join(PROJECT_ROOT, "knowledge", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    # 2. 使用真实 md 文件作为测试素材
    md_path = r"D:\code\shopkeeper_brain\temp_dir\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\hybrid_auto\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.md"
    print(f"测试 md: {md_path}")

    # 3. 运行节点完整流程
    from knowledge.processor.improt_processor.state import create_default_state
    from knowledge.processor.improt_processor.nodes.md_to_img_node import MarkdownToImageNode

    state = create_default_state(md_path=md_path)
    node = MarkdownToImageNode()
    print("\n===== 开始运行 MarkdownToImageNode =====")
    result = node(state)

    # 4. 输出结果
    print("\n===== 运行结果 =====")
    print(f"md_content 长度: {len(result.get('md_content', ''))}")
    summaries = result.get("image_summaries", {})
    print(f"image_summaries 数量: {len(summaries)}")
    for k, v in summaries.items():
        print(f"  - {k}: {v}")
    print("===== 测试完成 =====")







