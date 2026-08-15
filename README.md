# 掌柜智库 (ShopKeeper Brain)

基于 RAG（检索增强生成）的智能知识库问答系统。支持文档智能导入、混合向量检索、多路召回融合、智能重排序与流式问答，适用于产品手册、技术文档、企业知识库等场景。

## 功能特性

| 模块 | 描述 |
|------|------|
| 文档智能输入 | 支持 PDF/Markdown 文件上传，自动解析、切分、向量化 |
| 混合向量检索 | 稠密向量 + 稀疏向量（BM25）混合检索 |
| 多路召回融合 | 向量检索 + HyDE + Web 搜索 |
| 智能重排序 | Reranker 模型重排序，断崖检测动态截断 |
| 流式问答 | SSE 实时推送，逐 token 输出答案 |
| 会话历史管理 | MongoDB 存储对话历史，支持上下文连续对话 |

## 适用场景

- **产品手册问答** — 电子产品使用说明、维修手册等
- **技术文档检索** — API 文档、开发指南、FAQ 等
- **企业知识库** — 内部制度、操作规范、培训资料等
- **售后客服支持** — 产品故障排查、使用指导等

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  前端层 (Frontend Layer)              │
│    导入界面 (import.html)  │  聊天界面 (chat.html)   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│                   API 层 (API Layer)                  │
│  导入服务 :8000 (import_router.py)  POST /upload     │
│                                    GET  /status      │
│  查询服务 :8001 (query_router.py)   POST /query      │
│                                    GET  /stream      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│                 Processor 层                          │
│  导入处理流程 (import_process)  │  查询处理流程 (query_process) │
└─────────────────────────────────────────────────────┘
```

## 技术选型

| 类别 | 技术方案 | 说明 |
|------|----------|------|
| 后端框架 | FastAPI + Uvicorn | 异步高性能 HTTP 服务 |
| 工作流引擎 | LangGraph | 有状态图编排框架 |
| 大语言模型 | 阿里云 DashScope (Qwen) | qwen-flash / qwen3-vl-flash |
| 向量嵌入 | OpenAI API (text-embedding-v4) + BGE-M3 | 1536维 / 1024维+稀疏 |
| 重排序模型 | BGE-Reranker-Large | 本地部署 |
| 向量数据库 | Milvus | 混合检索（稠密+稀疏） |
| 文档数据库 | MongoDB | 对话历史存储 |
| 对象存储 | MinIO | 文件与图片存储 |
| PDF 解析 | MineRU | PDF 转 Markdown |
| 前端 | HTML5 + JS | 无框架，轻量实现 |

## 项目结构

```
shopkeeper_brain/
├── main.py                         # 应用入口
├── knowledge/
│   ├── .env                        # 环境变量配置（不提交）
│   ├── requirement.txt             # Python 依赖
│   ├── api/                        # API 路由层
│   │   └── __init__.py
│   ├── processor/
│   │   ├── __init__.py
│   │   ├── improt_processor/       # 文档导入处理流程（LangGraph）
│   │   │   ├── __init__.py
│   │   │   ├── main_graph.py       # 导入流程图定义
│   │   │   ├── config.py           # 导入流程配置管理
│   │   │   ├── state.py            # 图状态类型定义
│   │   │   ├── base.py             # 节点基类（统一接口/日志/异常）
│   │   │   ├── exceptions.py       # 自定义异常
│   │   │   └── nodes/              # 处理节点
│   │   │       ├── __init__.py
│   │   │       └── pdf_to_md_node.py  # PDF→Markdown 解析节点
│   │   └── query_processor/        # 查询处理流程
│   │       └── __init__.py
│   ├── service/                    # 业务服务层
│   │   └── __init__.py
│   ├── shcemal/                    # 数据模型（Pydantic Schema）
│   │   └── __init__.py
│   ├── prompt/                     # 测试用文档
│   │   └── 智能体-课程大纲.pdf
│   └── test/                       # 测试代码
│       ├── __init__.py
│       └── torchtest.py
├── .gitignore
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA GPU（推荐，用于 BGE 模型推理）
- [Milvus](https://milvus.io/) 向量数据库
- [MongoDB](https://www.mongodb.com/) 文档数据库
- [MinIO](https://min.io/) 对象存储

### 安装依赖

```bash
cd knowledge
pip install -r requirement.txt
```

### 配置环境变量

复制并编辑环境配置：

```bash
cp knowledge/.env.example knowledge/.env  # 或直接创建 .env 文件
```

关键配置项：

```bash
# LLM API（阿里云 DashScope）
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-flash

# BGE 本地模型路径
BGE_M3_PATH=/path/to/bge-m3
BGE_RERANKER_LARGE=/path/to/bge-reranker-large

# Milvus
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks_v1

# MongoDB
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=kb001

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### 启动服务

```bash
python main.py
```

## 核心设计

### LangGraph 工作流

文档导入流程采用 **LangGraph 有状态图** 编排，通过 `BaseNode` 基类统一节点接口：

- **状态管理**：`ImportGraphState`（TypedDict）定义全流程传递的数据结构
- **节点基类**：`BaseNode` 提供 `__call__` 统一入口，自动处理日志、任务追踪和异常包装
- **PDF 解析节点**：`PdfToMdNode` 调用 MinerU 子进程将 PDF 转为 Markdown，支持实时日志输出
- **配置中心**：`ImportConfig` 集中管理切片策略、模型参数、外部服务连接等配置

### 文档处理流水线

```
文件上传 → PDF解析(MineRU) → MD切分 → 图片提取&摘要 → 向量化(BGE-M3/OpenAI) → 入库(Milvus)
```

### 检索增强生成（RAG）流水线

```
用户问题 → 多路召回(向量+HyDE+Web) → 混合检索(稠密+稀疏) → Reranker重排序 → LLM生成 → SSE流式输出
```

## 依赖清单

```
minio                    # MinIO 对象存储客户端
langchain-openai         # OpenAI 兼容 LLM / Embedding
langgraph                # 工作流编排引擎
grandalf                 # 图布局渲染
pymilvus[model]          # Milvus 向量数据库客户端
sentence-transformers    # BGE 嵌入/重排序模型
pymongo                  # MongoDB 客户端
mineru[all]              # PDF 解析工具
openai-agents            # OpenAI Agents SDK
```

## License

MIT
