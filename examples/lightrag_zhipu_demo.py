# 导入必要的标准库
import os
import logging.config
import asyncio
import json
import numpy as np

# 从lightrag库中导入LightRAG类和QueryParam类
from lightrag import LightRAG, QueryParam
# 从lightrag.llm.zhipu模块中导入智普大模型的文本生成和嵌入函数
import lightrag.llm.zhipu as zhipu
# 从lightrag.utils模块中导入EmbeddingFunc类，用于定义嵌入函数
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
# 从lightrag.kg.shared_storage模块中导入初始化管道状态的函数
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv # new
load_dotenv()  # 这会自动加载当前目录下的 .env 文件
# 设置工作目录
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")  # 从环境变量获取工作目录，默认为"./rag_storage"
INPUT_DIR = os.getenv("INPUT_DIR", "./inputs")  # 从环境变量获取输入目录，默认为"./inputs"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

# 配置日志记录，设置日志格式和日志级别为INFO
# logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
# logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("zhipuai").setLevel(logging.DEBUG)

def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    # log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_dir = os.path.join(os.getcwd(), os.getenv("LOG_DIR","logs"))
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.DEBUG if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG" else logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")

# 如果工作目录不存在，则创建该目录
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
    
# 设置智普大模型的API密钥
#os.environ["ZHIPUAI_API_KEY"] = "812545cd88487a9a64e7d392406b2281.lJKQ0ys8c420uy70"

# 获取环境变量中的API密钥
# zhipu_api_key = os.environ.get("ZHIPUAI_API_KEY")
# # 如果API密钥未设置，则抛出异常提示用户设置API密钥
# if zhipu_api_key is None:
#     raise Exception("Please set ZHIPU_API_KEY in your environment")

# async def alinlp_embedding(
#     texts: list[str], 
#     size : str = "100",
#     api_type : str | None = None,
#     operation : str | None = None,
#     access_key_id: str | None = None, 
#     access_key_secret: str | None = None, 
#     **kwargs
# ) -> np.ndarray:
#     try:
#         from aliyunsdkcore.client import AcsClient
#         from aliyunsdkcore.request import CommonRequest
#     except ImportError:
#         raise ImportError("Please install aliyunsdkcore before initialize alinlp backend.")
#     if access_key_id and access_key_secret:
#         # 创建AcsClient实例
#         client = AcsClient(access_key_id,access_key_secret,"cn-hangzhou")
#     else:
#         # please set ALINLP key & secret in your environment
#         access_key_id = os.environ.get("ALINLP_AK_ENV")
#         access_key_secret = os.environ.get('ALINLP_SK_ENV')
#         if not access_key_id or not access_key_secret:
#             raise Exception(
#                 "Please set ALINLP_AK_ENV and ALINLP_SK_ENV in your environment"
#             )
#         client = AcsClient(access_key_id,access_key_secret,"cn-hangzhou")

#     request = CommonRequest()
#     # domain和version是固定值
#     request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
#     request.set_version('2020-06-29')

#     # action name可以在API文档里查到
#     request.set_action_name('GetWeChGeneral')

#     # 需要add哪些param可以在API文档里查到
#     request.add_query_param('ServiceCode', 'alinlp')
#     request.add_query_param('Size', size)
#     if api_type is not None:
#         request.add_query_param('Type', api_type) # 自动分词
#     if operation is not None:
#         request.add_query_param('Operation', operation) # 自动分词

#     # print(f"texts ================================= : {texts}")
#     # Convert single text to list if needed
#     if isinstance(texts, str):
#         texts = [texts]
    
#     len1 = len(texts)
#     # 去掉空字符串或去掉空格后为空的元素
#     texts = [t for t in texts if t and t.strip()]
#     len2 = len(texts)
#     if len1 != len2:
#         print(f"Removed {len1 - len2} empty texts.")

#     print(f"Embedding start ===========================")
#     embeddings = []
#     for text in texts:
#         try:
#             # print(f"Embedding text ================= : {text}")
#             request.add_query_param('Text', text)

#             # if text is None or text.isspace() or text == "":
#             #     print(f"Skipping empty text: {text}")
#             #     continue
#             # 打印完整的URL路径和参数
#             endpoint = f"https://{request.get_domain()}/"
#             params = request.get_query_params()
#             # 拼接参数字符串
#             param_str = "&".join([f"{k}={v}" for k, v in params.items()])
#             if not param_str:
#                 print(f"No query parameters found. text: {text}")
#             full_url = f"{endpoint}?{param_str}"
#             # print(f"Request URL: {full_url}")

#             response = client.do_action_with_exception(request)
#             resp_obj = json.loads(response)
#             data = json.loads(resp_obj["Data"])  # 先解析 Data 字符串
#             vec = data["result"]["vec"]          # 再取 vec 数组
#             # print(f"vector length: {len(vec)}")
#             embeddings.append(vec)            
#         except Exception as e:
#             logger.error(f"Error calling ALINLP Embedding API: {str(e)} - {full_url}")
#             print(f"Error calling ALINLP Embedding API: {str(e)} - {full_url}")
#             # 若API返回异常，填充一个全0的向量，长度为size参数
#             embeddings.append([0.0] * int[size])
#             #break

#     print(f"Embedding end ===========================")
#     return np.array(embeddings)


# 定义一个异步函数，用于初始化LightRAG实例
async def initialize_rag():
    # 创建LightRAG实例
    rag = LightRAG(
        working_dir=WORKING_DIR,  # 设置工作目录
        llm_model_func=zhipu.zhipu_complete_if_cache,  # 设置使用的语言模型函数
        llm_model_name=os.getenv("ZHIPUAI_LLM_MODEL","glm-4-flash-250414"),  # 设置使用的语言模型名称
        # llm_model_max_async=4,  # 设置最大异步请求数
        # llm_model_max_token_size=32768,  # 设置模型处理的最大token数量
        chunk_token_size=200, # 设置文本分块的token数量
        chunk_overlap_token_size=50,  # 设置文本分块的重叠token数量
        enable_llm_cache=os.environ.get("ENABLE_LLM_CACHE"),  # 启用LLM缓存
        enable_llm_cache_for_entity_extract=os.environ.get("ENABLE_LLM_CACHE_FOR_EXTRACT"),  # 启用实体提取的LLM缓存
        # 设置嵌入函数
        embedding_func=EmbeddingFunc(
            embedding_dim=50,  # 设置嵌入向量的维度 2048
            max_token_size=200,  # 设置模型处理的最大token数量 8192
            #func=lambda texts: zhipu_embedding(texts),  # 使用智普的嵌入函数
            func=lambda texts: zhipu.alinlp_embedding(
                texts=[t for t in texts if t and t.strip()], size="50"
            ),  # 使用 ali 的嵌入函数，过滤空字符串
        ),
    )

    # 初始化存储系统
    await rag.initialize_storages()
    # 初始化管道状态
    await initialize_pipeline_status()

    # 返回初始化好的LightRAG实例
    return rag

async def initialize_raganything(lightrag: LightRAG=None):
    # 首先，创建或加载现有的 LightRAG 实例
    # lightrag_working_dir = WORKING_DIR
    os.environ['PATH'] += f';{os.getenv("LIBREOFFICE_PATH")}'  # new

    # 检查是否存在之前的 LightRAG 实例
    if os.path.exists(WORKING_DIR) and os.listdir(WORKING_DIR):
        print("✅ Found existing LightRAG instance, loading...")
    else:
        print("❌ No existing LightRAG instance found, will create new one")

    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",  # 选择解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 现在使用现有的 LightRAG 实例初始化 RAGAnything
    if lightrag:
        ragany = RAGAnything(
            lightrag=lightrag,  # 传递现有的 LightRAG 实例
            config=config,
            vision_model_func=zhipu.zhipu_vision_model_func
            # 注意：working_dir、llm_model_func、embedding_func 等都从 lightrag_instance 继承
        )

        # 查询现有的知识库
        # result = await ragany.aquery_with_multimodal(
        #     "What data has been processed in this LightRAG instance?",
        #     mode="hybrid"
        # )
        # print("RAGAnything Query result:", result)
    else:
        ragany = RAGAnything(
            config=config,
            llm_model_func=zhipu.zhipu_complete_if_cache,
            vision_model_func=zhipu.zhipu_vision_model_func,
            embedding_func=zhipu.alinlp_embedding,
        )

    return ragany

# 定义主函数
async def main():
    # 使用异步事件循环初始化LightRAG实例
    print('initialize start ==========================================')
    is_clear_old_data = False
    is_rag_insert = True
    # rag = asyncio.run(initialize_rag())
    rag = await initialize_rag()
    ragany = await initialize_raganything(lightrag=rag)
    print('initialize end ==========================================')

    # 当前文件的绝对路径
    file_path = os.path.abspath(__file__)
    # 当前文件所属文件夹的绝对路径
    folder_path = os.path.dirname(file_path)
    # 当前文件所属文件夹的名称
    folder_name = os.path.basename(folder_path)

    # print("当前文件路径:", file_path)
    # print("所属文件夹路径:", folder_path)
    # print("所属文件夹名称:", folder_name)
    # print("doc path:=========="+os.getcwd()+"/"+folder_name+"/book.txt")

    try:
        # Clear old data files
        if is_clear_old_data:
            files_to_delete = [
                "graph_chunk_entity_relation.graphml",
                "kv_store_doc_status.json",
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
                "kv_store_llm_response_cache.json",
            ]

            for file in files_to_delete:
                file_path = os.path.join(WORKING_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleting old file:: {file_path}")

        if is_rag_insert:
            # file_path = os.getcwd()+"/"+folder_name+"/test-book.txt"
            folder_path = os.path.join(os.getcwd(), INPUT_DIR)
            print("Inserting start ..............................")
            print(f"folder path: {folder_path}")
            # 批量处理
            rag_result = await ragany.process_documents_with_rag_batch(
                file_paths=[folder_path],  # Process subset for demo
                output_dir=OUTPUT_DIR,
                parse_method="auto",
                max_workers=4,
                recursive=True,
                show_progress=True,
                backend="pipeline",          # 解析后端：pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client
                source="modelscope"        # 模型源："huggingface", "modelscope", "local"
            )

            print("\n" + "-" * 40)
            print("FULL RAG INTEGRATION RESULTS")
            print("-" * 40)
            print(f"Parse result: {rag_result['parse_result'].summary()}")
            print(
                f"RAG processing time: {rag_result['total_processing_time']:.2f} seconds"
            )
            print(
                f"Successfully processed with RAG: {rag_result['successful_rag_files']}"
            )
            print(f"Failed RAG processing: {rag_result['failed_rag_files']}")

            # 逐个文件处理
            # for filename in os.listdir(folder_path):
            #     file_path = os.path.join(folder_path, filename)

            #     if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
            #         print(f"Inserting file: {file_path}")
            #         # 读取本地文本文件并将其内容插入到LightRAG中
            #         with open(file_path, "r", encoding="utf-8") as f:
            #             await rag.ainsert(input=f.read(),
            #                     split_by_character="\n",  # 按照换行符进行分割
            #                     split_by_character_only=False
            #                     )
                        
            #     if os.path.isfile(file_path) and filename.lower().endswith('.docx'):
            #         # 处理文档
            #         await ragany.process_document_complete(
            #             file_path=file_path,
            #             output_dir=OUTPUT_DIR,
            #             parse_method="auto",
            #             backend="pipeline",          # 解析后端：pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client
            #             source="modelscope"        # 模型源："huggingface", "modelscope", "local"
            #         )

            #         # 查询处理后的内容
            #         # 纯文本查询 - 基本知识库搜索
            #         # text_result = await ragany.aquery(
            #         #     "文档的主要内容是什么？",
            #         #     mode="hybrid"
            #         # )
            #         # print("文本查询结果:", text_result)

            print("Inserting end ..............................")

        # 使用不同的查询模式进行检索并打印结果
        # 混合检索：混合模式，结合局部与全局信息
        question = "3HK postpaid eSIM 流程中，客人需要在哪一步上传HKID？"

        print("Querying with different modes - hybrid:")
        print(
            await rag.aquery(
                question, param=QueryParam(mode="hybrid")
            )
        )

        # 朴素检索：基础查询模式
        print("Querying with different modes - naive:")
        print(
            await rag.aquery(
                # "故事中的人物关系有哪些?", param=QueryParam(mode="naive")
                question, param=QueryParam(mode="naive")
            )
        )

        # 局部检索：局部检索模式，仅查找与输入相关的区域
        print("Querying with different modes - local:")
        print(
            await rag.aquery(
                question, param=QueryParam(mode="local")
            )
        )

        # 全局检索：全局检索模式，扩展到整个知识图谱的关系
        print("Querying with different modes - global:")
        print(
            await rag.aquery(
                question, param=QueryParam(mode="global")
            )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await rag.finalize_storages()


async def test_zhipu_complete():
    from zhipuai import ZhipuAI
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("zhipuai").setLevel(logging.DEBUG)

    # import httpx
    # import certifi

    # # 使用 certifi 提供的 CA 证书路径
    # ca_cert_path = certifi.where()
    # #ca_cert_path = r"C:\d\Program Files\workspace\cert\bigmodel_cn.crt"
    # print(f"Using CA certs from: {ca_cert_path}")

    # client = httpx.Client(verify=ca_cert_path)
    # response = client.get("https://open.bigmodel.cn/api/paas/v4/chat/completions")
    # print(response)

    client = ZhipuAI(api_key="812545cd88487a9a64e7d392406b2281.lJKQ0ys8c420uy70")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        # model="glm-4-plus",  # 请填写您要调用的模型名称
        model="glm-4-flash-250414",
        messages=[
            # {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
            # {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
            # {"role": "user", "content": "智谱AI开放平台"},
            # {"role": "assistant", "content": "点燃未来，智谱AI绘制无限，让创新触手可及！"},
            # {"role": "user", "content": "创作一个更精准且吸引人的口号"}
            {"role": "user", "content": "故事中分别有哪些人物?"},
        ],
    )
    print(response) 



# 如果当前模块是主模块，则调用主函数
if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    asyncio.run(main())
    # asyncio.run(test_zhipu_complete())

    print("\nDone!")

    
    



