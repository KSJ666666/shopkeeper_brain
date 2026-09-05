import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_upload_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse
from knowledge.service.upload_service import UploadService


def create_route(app:FastAPI):
    """
    注册fastapi路由，相当于添加web接口
    :return:
    """
    @app.get("/")
    def hello_fastapi():
        return {
            "hello": "fastapi"
        }
    #上传请求
    @app.post("/upload",response_model=UploadResponse)
    def upload_endpoint(file: UploadFile ,
                        background_tasks:BackgroundTasks,
                        upload_service:UploadService = Depends(get_upload_file_service),
                        ):
        """
        处理文件的上传
        :param file: 上传的文件
        :param upload_service: 上传服务
        :return: 上传响应
        """
        #1.讲上传的文件存入本地临时目录并存入远程minio
        task_id,import_file_path,file_dir = upload_service.process_upload_file(file)

        #2.运行整个节点的导入图谱
        # 这里pdf转md解析时间很长，在此处设置定时任务，在后台异步处理
        background_tasks.add_task(upload_service.run_import_graph, task_id, import_file_path, file_dir)

        #3.返回响应数据
        return UploadResponse(message= f"文件:{file.filename}上传成功，",task_id=task_id)


def create_fastapi()->FastAPI:
    """
    创建fastapi示例并返回
    :return:
    """
    #1.创建fastapi实例，
    app = FastAPI(description="shopkeeper_fastapi实例")
    #注册路由\

    #2.配置CORS跨域资源共享

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ← 和 credentials=True 冲突
        allow_credentials=False,  # 自定义cookies Authorization tsl客户端证书信息
        allow_methods=["*"],  # ← 和 credentials=True 冲突 GET(获取资源) POST(新增) DELETE(删除) PUT(修改)
        allow_headers=["*"],  # ← 和 credentials=True 冲突 自定义的头字段 token content‑type:application/json
    )
    #3.挂载静态资源
    page_dir = get_front_page_dir()
    app.mount("/front",StaticFiles(directory=page_dir))


    create_route(app)

    return app


if __name__ == '__main__':
    """
    :param1:  fastapi 示例
    :param2:  服务ip地址
    :param3:  端口号
    """
    uvicorn.run(app=create_fastapi(),host="0.0.0.0",port=8000)