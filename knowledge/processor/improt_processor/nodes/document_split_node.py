# -*- coding: utf-8 -*-
"""
document_split_node.py 逐行注释版
用途：把一篇 Markdown 文档按标题层级切分成多个"章节"（section），供后续向量化/入库使用。
说明：本文档在保留原逻辑的基础上补充注释，并标注了原有代码存在的问题（见 BUG 标记）。
"""
import re
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.improt_processor import config
from knowledge.processor.improt_processor.base import BaseNode, setup_logging, T
from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.processor.improt_processor.exceptions import StateFieldErro,  r, ValidationError


class DocumentSplitNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分的核心逻辑入口
        预期流程：
            1. 从 state 校验并取出 md_content / file_title / 长度配置
            2. 按标题把 md_content 切分为多个 section
            3. 把 sections 写回 state
        """

        config = self.config
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)

        # 2. 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:
        """
        参数校验：从 state 中取出文档内容与标题，统一换行符，并校验切片长度配置。
        """
        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符：把 \r\n / \r 全部归一为 \n，避免不同平台换行导致切分错乱
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值：长度参数必须为正，且最大值必须大于最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        根据 Markdown 标题（# ~ ######）把文档切成多个章节。
        每个章节包含：
            body:         正文内容（该标题之下、下一标题之前的所有行）
            title:        当前标题
            parent_title: 父标题（用于后续短章节合并时判断"是否同源"）
            file_title:   整个文档的标题
        """
        in_fence = False    # 是否处于代码块内（``` 或 ~~~），代码块里的 # 不算标题
        body_liens = []     # 暂存当前章节收集到的正文行
        sections = []       # 最终收集到的章节对象
        current_title = ""  # 当前章节标题
        hierarchy = [""] * 7  # 标题层级追踪数组：hierarchy[i] = 最近遇到的第 i 级标题
        current_level = 0     # 当前标题的层级（# 的个数）

        def _flush() -> List[Dict[str, Any]]:
            """
            把暂存的行打包成一个 section 对象。
            打包规则：
              - current_title 为空、body 有内容  → 可打包（作为文档开头的导语等，有意义）
              - current_title 有、body 为空      → 可打包（空章节标题，保留以备合并）
              - 两者都有                        → 一定打包
              - 两者都为空                      → 不打包
            """
            # 1. 将收集到的正文行拼接成字符串
            body = "\n".join(body_liens)

            if current_title or body:
                parent_title = ""
                # 找父标题：从当前层级的上一级开始，向上找最近一个"非空"的层级标题
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:          # 跳过空值，读到最近的父标题
                        parent_title = hierarchy[i]
                        break

                # 兜底：找不到父标题时，用当前标题或文档标题作为父标题
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,  # 内容标题
                    "parent_title": parent_title,  # 内容父标题
                    "file_title": file_title,      # 文档标题
                })

        # 1. 按换行符把整篇 md 切成行
        md_lines = md_content.split("\n")

        # 2. 标题正则：行首允许空白，后跟 1~6 个 # 和至少一个空格
        #    group(1) = # 的个数（层级），group(2) = 标题内容
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 逐行扫描
        for md_line in md_lines:

            # 3.1 检测代码块边界（``` 或 ~~~），遇到边界就翻转状态
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence  # 翻转而非固定赋值，保证嵌套/进出配对正确

            # 3.2 不在代码块内时才尝试匹配标题正则
            match = heading_re.match(md_line) if not in_fence else None

            # 3.3 匹配到了标题（且一定不是代码块里的 #）
            if match:
                # 先把前面暂存的行打包成 section（结束上一个章节）
                _flush()

                current_title = md_line               # 记录当前标题（整行，含 #）
                level = len(match.group(1))           # 标题层级 = # 的个数
                current_level = level                 # 更新当前层级
                hierarchy[level] = current_title      # 写入当前层级的标题

                # 清空所有比当前层级更低的层级记录（表示新开分支）
                for i in range(level + 1, 7):
                    hierarchy[i] = ""

            # 没匹配到标题（普通行）或处于代码块内 → 收集进暂存区
            else:
                body_liens.append(md_line)

        # 收尾：把最后一段暂存的行也打包
        _flush()  # BUG 3: 原文件此处写成 _flush（缺括号），只引用函数不执行，需改为 _flush()
        return sections
