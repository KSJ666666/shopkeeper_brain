from dataclasses import dataclass
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


