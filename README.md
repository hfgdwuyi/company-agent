# company-rag — 企业级 RAG 知识库 + Agent 系统

基于本地 **Docker TensorRT-LLM（Qwen2.5-7B INT4）** 推理的企业级 RAG 知识库与智能体，五大能力：

1. **文档解析**：PDF / DOCX / TXT / MD（扫描件 OCR 可选）
2. **RAG 向量检索**：bge-small-zh-v1.5 中文嵌入 + ChromaDB 持久化向量库
3. **TensorRT-LLM 本地推理**：对接 `trt-llm-qwen` 容器的 OpenAI 兼容 API（`127.0.0.1:8001`）
4. **Agent 工具调用**：原生 tool-calling 协议 + 工具注册表（知识检索 / 计算 / 时间 / 文档管理）
5. **FastAPI 对外接口**：REST + SSE 流式 + Swagger 文档

```
┌───────────────────────────────────────────────────────────────┐
│  FastAPI 服务 (0.0.0.0:8010)                                   │
│  /api/v1/documents/upload · /query · /chat · /tools · /health │
│                                                               │
│  解析器 → 分块器 → 嵌入器(bge-small-zh) → 向量库(ChromaDB)      │
│        ↓                                                      │
│  RAG 管线 ──┐                                                 │
│  Agent ─────┴──► TRT-LLM 客户端 ──► Docker: trt-llm-qwen:8000 │
│                    (工具选择→具名tool_choice→执行→回传)          │
└───────────────────────────────────────────────────────────────┘
```

## 目录结构

```
C:\AI\company-rag\
├── src\company_rag\
│   ├── config.py           # 配置（RAG_* 环境变量覆盖）
│   ├── models.py           # Pydantic API Schema
│   ├── document_parser.py  # PDF/DOCX/TXT/MD 解析
│   ├── chunker.py          # 中文递归分块（句边界优先 + overlap）
│   ├── embedder.py         # bge-small-zh-v1.5 嵌入（CPU，不抢显存）
│   ├── vector_store.py     # ChromaDB 持久化 + 余弦检索
│   ├── llm_client.py       # TRT-LLM OpenAI 兼容客户端（工具调用协议封装）
│   ├── rag.py              # RAG 检索 + 合成（带来源引用）
│   ├── tools.py            # 工具注册表（retrieve_knowledge/calculator/…）
│   ├── agent.py            # 两阶段工具调用 Agent（决策轮 + 强制具名轮）
│   ├── knowledge_base.py   # 文档生命周期（入库/删除/清单）
│   └── api.py              # FastAPI 应用
├── scripts\
│   ├── ingest.py           # CLI：批量入库/清单/删除
│   ├── download_corpus.py  # 下载真实测试语料
│   ├── demo.py             # 控制台演示（RAG + Agent）
│   └── probe_*.py          # TRT-LLM 端点探测脚本
├── data\
│   ├── raw\                # 原始文档（真实 PDF 等）
│   ├── parsed\             # 上传临时目录
│   └── chroma\             # 向量库持久化
├── models\                 # 嵌入模型（ModelScope 下载）
├── deploy\                 # Dockerfile + docker-compose.yml
└── tests\test_smoke.py     # 冒烟测试
```

## 快速开始

### 0. 前置条件

- Docker 中已运行 TensorRT-LLM 服务（本仓库对接 `trt-llm-qwen` 容器，宿主端口 **8001**）：
  ```
  docker ps   # 应看到 trt-llm-qwen 容器，0.0.0.0:8001->8000/tcp
  ```
- Python 3.10+（本机 3.13，miniconda）

### 1. 安装依赖

```powershell
cd C:\AI\company-rag
pip install -e .          # 或按需: pip install pypdf chromadb sentence-transformers python-multipart fastapi uvicorn httpx
```

### 2. 下载嵌入模型（ModelScope，hf-mirror 备选）

```powershell
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir=r'C:\AI\company-rag\models')"
# 未安装 modelscope 时，代码会自动尝试 hf-mirror 下载
```

### 3. 下载真实测试语料（arXiv 论文 + 国新办白皮书实录 + 法规全文）

```powershell
python scripts/download_corpus.py
```

### 4. 批量入库

```powershell
python scripts/ingest.py data\raw          # 入库全部文档
python scripts/ingest.py --list            # 查看清单
python scripts/ingest.py --stats           # 统计
python scripts/ingest.py --delete <doc_id> # 删除
```

### 5. 启动服务

```powershell
python -m uvicorn company_rag.api:app --host 0.0.0.0 --port 8010
# Swagger: http://127.0.0.1:8010/docs
```

### 6. 控制台演示（不起 HTTP）

```powershell
python scripts/demo.py
# 输入 "r:问题" 走纯 RAG；其它输入走 Agent（自动决定是否调工具）
```

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息（端点清单） |
| GET | `/ui` | **用户查询页面**（浏览器直接搜文档/RAG 问答/Agent 对话） |
| GET | `/api/v1/health` | 健康检查：LLM 连通性 + 知识库状态 |
| POST | `/api/v1/documents/upload` | 上传文档（multipart，自动解析+入库） |
| GET | `/api/v1/documents` | 文档清单 |
| GET | `/api/v1/documents/{doc_id}` | 文档详情（片段预览） |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其向量 |
| POST | `/api/v1/search` | **文档检索**（向量检索，不调 LLM，返回命中片段+来源+相似度，支持 `doc_ids` 过滤 / `min_score` 阈值） |
| POST | `/api/v1/query` | RAG 问答（返回答案 + 来源引用） |
| POST | `/api/v1/query/stream` | RAG 流式（SSE） |
| POST | `/api/v1/chat` | Agent 对话（自动工具调用） |
| POST | `/api/v1/chat/stream` | Agent 流式对话（SSE，工具事件 + token 流） |
| GET | `/api/v1/tools` | 工具清单（OpenAI schema） |
| POST | `/api/v1/tools/invoke` | 直接调用工具（调试） |

### 示例

```bash
# 文档检索（用户直接查询知识库，不调 LLM，秒回）
curl -X POST http://127.0.0.1:8010/api/v1/search -H "Content-Type: application/json" \
  -d '{"query": "数据安全 数据处理义务", "top_k": 5, "min_score": 0.3}'

# RAG 问答
curl -X POST http://127.0.0.1:8010/api/v1/query -H "Content-Type: application/json" \
  -d '{"question": "Attention is All You Need 论文提出了什么结构？", "top_k": 3}'

# Agent 对话（自动工具调用）
curl -X POST http://127.0.0.1:8010/api/v1/chat -H "Content-Type: application/json" \
  -d '{"message": "知识库里有哪几份文档？顺便算一下 128*0.15/4 等于多少"}'

# 上传文档
curl -X POST http://127.0.0.1:8010/api/v1/documents/upload \
  -F "file=@data\raw\attention_is_all_you_need.pdf"

# 用户查询页面（浏览器打开）
# http://127.0.0.1:8010/ui
```

鉴权：设置环境变量 `RAG_API_KEY=xxx` 后，所有 `/api/v1/*` 需带 `Authorization: Bearer xxx`。

## Agent 工具调用设计（对接 TRT-LLM 的实测协议）

TRT-LLM 0.21 `trtllm-serve` 的 OpenAI 接口工具调用有三个实测限制：

1. `tool_choice` 只支持**具名函数**（`auto`/`required` 会 400）；
2. 工具参数以 `<tool_call>…</tool_call>` 包裹的 JSON 返回；
3. `tool` 角色回传消息**不能带 `name` 字段**（否则 400）。

因此 Agent 采用两阶段协议：

```
用户问题
  │
  ▼
决策轮: 模型从工具清单中选工具（JSON）或直接回答
  │ 需要工具
  ▼
强制轮: tool_choice=<具名工具> 走服务端原生 tool-calling，获得结构化参数
  │
  ▼
执行工具 → 以 tool 角色消息回传结果 → 回到决策轮（≤5 轮）
  │ 无需工具
  ▼
最终回答（可流式）
```

工具清单：`retrieve_knowledge`（RAG 检索）、`calculator`（安全数学）、`get_current_datetime`、`list_documents`、`get_document_summary`、`count_documents`。

## Docker 部署

```powershell
docker compose -f deploy\docker-compose.yml up -d --build   # 首次构建
docker compose -f deploy\docker-compose.yml up -d           # 日常启动（复用已有镜像）
```

- 容器内通过 `host.docker.internal:8001` 访问宿主机 TRT-LLM；
- 嵌入模型与 Chroma 数据目录挂载自宿主（数据持久化）；
- **代码热挂载**：宿主 `src\company_rag` 直接挂载覆盖容器内包目录——
  - 改网页 `src\company_rag\web\index.html`：**刷新即生效**（无需重启/重建）；
  - 改 Python 代码：`docker compose restart company-rag`（几秒）；
  - **镜像仅在首次部署或依赖变更时需要重建**（`--build`），日常迭代零重建；
- 对外暴露 `0.0.0.0:8010`，建议设置 `RAG_API_KEY` 并收敛到网关。

## 配置项（环境变量，前缀 RAG_）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_DATA_DIR` | `data` | 数据根目录（含 images 图片库） |
| `RAG_LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | TRT-LLM OpenAI 端点 |
| `RAG_LLM_MODEL` | `engine_qwen_int4_v2` | 模型名（不校验） |
| `RAG_EMBED_MODEL_PATH` | `models\bge-small-zh-v1.5` | 嵌入模型目录 |
| `RAG_EMBED_DEVICE` | `cpu` | 嵌入设备（避免抢占推理显存） |
| `RAG_CHROMA_DIR` | `data\chroma` | 向量库持久化目录 |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `220` / `50` | 分块参数（受输入预算约束，宜小） |
| `RAG_TOP_K` | `3` | 默认检索条数 |
| `RAG_LLM_MAX_INPUT_TOKENS` | `430` | 决策/检索输入 token 预算（上限 448） |
| `RAG_API_KEY` | 空 | 非空则启用 Bearer 鉴权 |
| `RAG_AGENT_MAX_ITERATIONS` | `5` | Agent 最大工具轮数 |
| `RAG_LLM_MAX_INPUT_TOKENS` | `430` | 输入 token 预算（引擎上限 448） |

## 图表展示

PDF 入库时用 **pymupdf** 自动提取页面插图（过滤装饰性小图），保存到 `data/images/{doc_id}/page_xxx/`。RAG 回答与检索结果会附带图片 URL（`/api/v1/images/...`），UI 页面自动展示——例如问"Transformer 架构"，回答下方会直接显示论文图1 架构图。

## 测试

```powershell
python tests\test_smoke.py   # 分块/计算器/工具参数解析/向量库 CRUD 冒烟测试
```

## 在另一台设备部署（GitHub 克隆后）

仓库不包含引擎/模型/语料（均已 `.gitignore`），新设备按以下步骤自备：

```powershell
# 1. 克隆并装依赖
git clone <repo-url> C:\AI\company-rag && cd C:\AI\company-rag
pip install -e .
pip install pymupdf modelscope

# 2. 准备 TRT-LLM 引擎（核心前提）
#    - 用 NVIDIA TensorRT-LLM 0.21 镜像，参考 scripts/probe_trtllm.py 的协议；
#    - 在目标机重新量化+编译引擎（引擎绑定 GPU 架构 sm_XX，不能跨机拷贝）：
#      容器内: python /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
#        --model_dir <hf Qwen2.5-7B-Instruct> --output_dir ckpt --dtype float16 \
#        --use_weight_only --weight_only_precision int4 --load_model_on_cpu
#      trtllm-build --checkpoint_dir ckpt --output_dir engine --max_batch_size 8 \
#        --max_input_len 4096 --max_seq_len 4096 --max_num_tokens 8192 --gemm_plugin float16 ...
#    - 启动: trtllm-serve serve engine --tokenizer <hf dir> --host 0.0.0.0 --port 8000
#    - 或在 Docker Desktop 里挂载上述引擎目录，宿主端口 8001（见 deploy/docker-compose.yml）
#    - 注意：8GB 卡上 KV cache 预算约 449 tokens/请求（输入+输出），回答会偏短（详见"已知限制"）

# 3. 下载嵌入模型（ModelScope）
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir=r'C:\AI\company-rag\models')"

# 4. 下载真实语料并入库
python scripts/download_corpus.py
python scripts/ingest.py data\raw

# 5. 启动
docker compose -f deploy\docker-compose.yml up -d    # 或: uvicorn company_rag.api:app --port 8010
```

> 引擎/模型/语料均可重建，仓库保持轻量（约 500KB 纯代码）。

## 性能评估

内置两套评估脚本（真实 TRT-LLM + 已入库语料）：

```powershell
python scripts/evaluate_rag.py     # RAG：检索质量 + 生成质量
python scripts/evaluate_agent.py   # Agent：工具调用正确率
```

**评估维度与指标：**

| 维度 | 指标 | 实测（12 份真实文档） |
|------|------|---------------------|
| 检索质量（无 LLM） | Recall@1 / @3 / @5 / MRR@10 | 73.6% / 93.1% / 95.8% / 83.6% |
| 生成质量（LLM-as-judge，本地 Qwen 评分） | 忠实度 faithfulness / 相关性 relevance（1-5） | 4.1 / 4.5 |
| Agent 能力（8 任务集断言） | 工具选择准确率 / 回答达标率 / 通过率 | 100% / 100% / 8-8 |

- 检索评估：从每份文档抽取片段构造"查询→应命中片段"测试集（72 条），自动统计 Recall/MRR；
- 生成评估：10 个真实问答对 → RAG 回答 → 本地 Qwen 当裁判打忠实度/相关性分（受 448 token 预算约束，judge 上下文自动截短）；
- Agent 评估：8 个任务（计算器/知识检索/文档清单/时间/统计/纯对话），断言期望工具、参数与回答关键词。

报告自动保存到 `results/rag_eval_report.json` 与 `results/agent_eval_report.json`。

## 已知限制

- **8GB 显存（RTX 4070 Laptop）**：TRT-LLM 引擎加载后剩余显存有限，实测每请求最大输入约 **448 tokens**（KV cache 约束，非引擎 `max_seq_len=4096`）。系统已内置 token 预算感知：
  - RAG 上下文按预算动态装载（约 2-3 个短块）；
  - Agent 决策/工具回传/最终回答均按预算截断与裁剪历史；
  - 若需更大上下文，可重启 `trt-llm-qwen` 容器时调小 `--max_batch_size`（如 8→2），输入上限约提升 4 倍（约 1792 tokens）；
- TRT-LLM 单卡约 18~21 tokens/s，建议同时在线 ≤8 人；
- 扫描版 PDF 需启用 OCR（安装 `pytesseract` + `pdf2image` + tesseract 语言包，并设 `RAG_SCAN_PDF_OCR=1`）；
- 嵌入式向量库 ChromaDB 适合中小规模（百万级片段内）；更大规模建议迁移 Qdrant/Milvus（`vector_store.py` 接口可替换）。
- **复合问题**（一条消息含多个独立子任务，如"有哪几份文档？再算个 2*3"）：当前 Qwen2.5-7B INT4 在决策轮基本只选一个工具或直接作答，多工具链式调用不稳定——单意图工具调用（计算/检索/清单/时间）已实测可靠。可通过更强的模型（如 32B+）或改为"先拆分子问题再逐个路由"的上层编排来增强（`agent.py` 的 `detect_intents` 已预留意图识别入口）。

## 实测结果（真实语料 + 本地 TRT-LLM）

已入库 **12 份真实文档**（8 篇 arXiv 论文 + 数据安全法/个人信息保护法全文 + 白皮书实录），共约 4200 个向量片段：

| 场景 | 结果 |
|------|------|
| RAG：英文论文问答（Transformer 核心创新） | ✅ 正确，带来源页码 |
| RAG：中文法规问答（数据安全法原则） | ✅ 正确，带来源引用 |
| RAG：RAG 论文问答 | ✅ 正确 |
| Agent：计算器工具 `(128.5*3+75.25)/4` | ✅ 调工具得 115.1875，附推导 |
| Agent：知识检索（个人信息保护法） | ✅ 检索+归纳七大原则 |
| Agent：文档清单 / 当前时间 | ✅ 基于工具结果作答 |
| FastAPI 全端点（含 SSE 流式、上传/删除、鉴权 401/200） | ✅ 通过 |
