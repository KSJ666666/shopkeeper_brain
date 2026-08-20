import re
from typing import Tuple, List, Dict, Any



class DocumentSplitNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分的核心逻辑入口
        Args:
            state:

        Returns:

        """
        # 1. 参数校验
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state)

        # 2. 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:

        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:

        """
        parent_title:封装的原因主要为了后面短section在合并的时候有一个判断标准（同源：同一个父标题）
        根据标题来切分（# {1,6}都有可能）
        Args:
            md_content: 切分的md
            file_title: 上传文档标题

        Returns:
         List[Dict]:切分后的多个章节
        """

        in_fence = False  # 是否在代码块内
        body_liens = []
        sections = []  # 最终收集到的章节对象
        current_title = ""
        hierarchy = [""] * 7  # （数组）存储所有标题内容（作为section的父标题使用） 标题层级追踪数组
        current_level = 0

        def _flush() -> List[Dict[str, Any]]:
            """
            打包section
            {
            "body": "收集到的所有行"
            “title”:"当前内容的标题"
            "parent_title":当前内容的父标题（最麻烦）
            "file_title":文档标题（最简单）
            }
            Returns:
            如果current_title没有，body有 能进入打包成section,【也有意义】
            如果current_title有,body没有。也打包成section:在合并阶段可以保留上【可选：建议留下来】在后续合并阶段没有任何影响
            如果current_title有，body也有 能进入打包成section【一定留】
            如果current_title没有 body也没有 不会进入（不能打包）
            """

            # 1. 处理内容行
            body = "\n".join(body_liens)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:  # 找父标题的时候 排除某一个位置的空值
                        parent_title = hierarchy[i]  # 读取操作
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,  # 内容标题
                    "parent_title": parent_title,  # 内容父标题
                    "file_title": file_title,
                })

        # 1. 根据\n切分md_content
        md_lines = md_content.split("\n")

        # 2. 定义正则（正则的规则是从MD中找标题#{1,6}）():捕获组:产生三个group(0) group(1):#(1)#(6) group(2)标题的内容
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 遍历切分后md_lines
        for md_line in md_lines:

            # 3.1 检测代码块边界（``` 或 ~~~）代码块要留下来
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence  # 不要用固定true  or false

            # 3.2 判读是否要走正则
            match = heading_re.match(md_line) if not in_fence else None

            # 3.3 判断math 是否有
            # 代表匹配到了标题而且一定是非代码块中的# 标题
            if match:

                # 将 body_liens中收集到的行封装到section对象
                _flush()

                current_title = md_line  # 当前标题
                level = len(match.group(1))  # 当前标题的层级（# {1,6}）
                current_level = level
                hierarchy[level] = current_title  # 写入操作

                for i in range(level + 1, 7):
                    hierarchy[i] = ""  # 下面的清空
                # 没有匹配到标题[普通行] 或者是代码块（加入）
            else:
                body_liens.append(md_line)

        _flush()
        return sections



