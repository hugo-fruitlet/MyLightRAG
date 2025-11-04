import sys
import re
import json
from ..utils import verbose_debug

if sys.version_info < (3, 9):
    pass
else:
    pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("zhipuai"):
    pm.install("zhipuai")

from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)

from lightrag.types import GPTKeywordExtractionFormat

import numpy as np
from typing import Union, List, Optional, Dict

import os
from dotenv import load_dotenv # new
load_dotenv()  # 这会自动加载当前目录下的 .env 文件

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def zhipu_complete_if_cache(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = os.getenv("ZHIPUAI_LLM_MODEL","glm-4-flash-250414"),  # The most cost/performance balance model in glm-4 series
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    **kwargs,
) -> str:
    # dynamically load ZhipuAI
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError("Please install zhipuai before initialize zhipuai backend.")

    if api_key:
        client = ZhipuAI(api_key=api_key)
    else:
        # please set ZHIPUAI_API_KEY in your environment
        api_key = os.environ["ZHIPUAI_API_KEY"]
        if api_key is None:
            raise Exception("Please set ZHIPU_API_KEY in your environment")
        client = ZhipuAI(api_key=api_key)

    messages = []

    if not system_prompt:
        system_prompt = "You are a helpful assistant. Note that sensitive words in the content should be replaced with ***"

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Add debug logging
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    # Remove unsupported kwargs
    kwargs = {
        k: v for k, v in kwargs.items() if k not in ["hashing_kv", "keyword_extraction"]
    }
    print("===== start zhipu_complete_if_cache =====")
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    print("===== end zhipu_complete_if_cache =====")
    print("===== Query Output from LLM =====")

    # 兼容 streaming (StreamResponse) 和 非 streaming 返回
    content_text = ""
    try:
        # 流式响应：包含 _stream_chunks 生成器（按块返回 ChatCompletionChunk）
        if hasattr(response, "_stream_chunks"):
            for chunk in response._stream_chunks:  # generator of ChatCompletionChunk
                try:
                    # chunk 可能是对象或 dict，优先尝试常见字段
                    if hasattr(chunk, "choices"):
                        for ch in chunk.choices:
                            # delta 或 message 两种可能结构
                            if hasattr(ch, "delta") and getattr(ch.delta, "content", None):
                                content_text += ch.delta.content
                            elif hasattr(ch, "message") and getattr(ch.message, "content", None):
                                content_text += ch.message.content
                            elif isinstance(ch, dict):
                                content_text += ch.get("delta", {}).get("content", "") or ch.get("message", {}).get("content", "")
                    else:
                        # 兜底：尝试把 chunk 转为 dict 并拼接可用字符串字段
                        cdict = getattr(chunk, "__dict__", None) or (chunk if isinstance(chunk, dict) else None)
                        if isinstance(cdict, dict):
                            for v in cdict.values():
                                if isinstance(v, str):
                                    content_text += v
                except Exception:
                    # 单块解析失败则跳过
                    continue

        # 非流式响应：直接包含 choices
        elif hasattr(response, "choices"):
            try:
                content_text = response.choices[0].message.content
            except Exception:
                # 有些实现可能返回 dict 风格
                if hasattr(response, "to_dict"):
                    d = response.to_dict()
                    content_text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    content_text = str(response)

        # 其它可序列化对象：to_dict -> json
        elif hasattr(response, "to_dict"):
            d = response.to_dict()
            content_text = json.dumps(d, ensure_ascii=False)

        else:
            content_text = str(response)
    except Exception as e:
        content_text = f"[error extracting content] {e}"

    # print(content_text)
    logger.debug(f"Query Output from LLM: {content_text}")


    # 返回解析出的文本（和之前语义一致）
    return content_text
    # return response.choices[0].message.content


async def zhipu_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    # Pop keyword_extraction from kwargs to avoid passing it to zhipu_complete_if_cache
    keyword_extraction = kwargs.pop("keyword_extraction", None)

    if keyword_extraction:
        # Add a system prompt to guide the model to return JSON format
        extraction_prompt = """You are a helpful assistant that extracts keywords from text.
        Please analyze the content and extract two types of keywords:
        1. High-level keywords: Important concepts and main themes
        2. Low-level keywords: Specific details and supporting elements

        Return your response in this exact JSON format:
        {
            "high_level_keywords": ["keyword1", "keyword2"],
            "low_level_keywords": ["keyword1", "keyword2", "keyword3"]
        }

        Only return the JSON, no other text."""

        # Combine with existing system prompt if any
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{extraction_prompt}"
        else:
            system_prompt = extraction_prompt

        try:
            response = await zhipu_complete_if_cache(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

            # Try to parse as JSON
            try:
                data = json.loads(response)
                return GPTKeywordExtractionFormat(
                    high_level_keywords=data.get("high_level_keywords", []),
                    low_level_keywords=data.get("low_level_keywords", []),
                )
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract JSON from text
                match = re.search(r"\{[\s\S]*\}", response)
                if match:
                    try:
                        data = json.loads(match.group())
                        return GPTKeywordExtractionFormat(
                            high_level_keywords=data.get("high_level_keywords", []),
                            low_level_keywords=data.get("low_level_keywords", []),
                        )
                    except json.JSONDecodeError:
                        pass

                # If all parsing fails, log warning and return empty format
                logger.warning(
                    f"Failed to parse keyword extraction response: {response}"
                )
                return GPTKeywordExtractionFormat(
                    high_level_keywords=[], low_level_keywords=[]
                )
        except Exception as e:
            logger.error(f"Error during keyword extraction: {str(e)}")
            return GPTKeywordExtractionFormat(
                high_level_keywords=[], low_level_keywords=[]
            )
    else:
        # For non-keyword-extraction, just return the raw response string
        return await zhipu_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
def zhipu_vision_model_func(
        prompt, system_prompt=None, history_messages=[], type=None, url=None, **kwargs
    ):
        print("===== start zhipu_vision_model_func =====")
        if type == "image_url":
            return zhipu_complete_if_cache(
                model = os.getenv("ZHIPUAI_VLLM_MODEL","glm-4.1v-thinking-flash"),
                prompt="",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{url}"
                                },
                            },
                        ],
                    }
                    if url
                    else {"role": "user", "content": prompt},
                ],
                # api_key=api_key,
                # base_url=base_url,
                **kwargs,
            )
        else:
            return zhipu_vision_model_func(prompt, system_prompt, history_messages, **kwargs)

@wrap_embedding_func_with_attrs(embedding_dim=1024)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def zhipu_embedding(
    texts: list[str], model: str = "embedding-3", api_key: str = None, **kwargs
) -> np.ndarray:
    # dynamically load ZhipuAI
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError("Please install zhipuai before initialize zhipuai backend.")
    if api_key:
        client = ZhipuAI(api_key=api_key)
    else:
        # please set ZHIPUAI_API_KEY in your environment
        # os.environ["ZHIPUAI_API_KEY"]
        client = ZhipuAI()

    # Convert single text to list if needed
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(model=model, input=[text], **kwargs)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            raise Exception(f"Error calling ChatGLM Embedding API: {str(e)}")

    return np.array(embeddings)

@wrap_embedding_func_with_attrs(embedding_dim=1024)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def alinlp_embedding(
    texts: list[str], 
    size : str | None = None,
    api_type : str | None = None,
    operation : str | None = None,
    access_key_id: str | None = None, 
    access_key_secret: str | None = None, 
    **kwargs
) -> np.ndarray:
    try:
        from aliyunsdkcore.client import AcsClient
        from aliyunsdkcore.request import CommonRequest
    except ImportError:
        raise ImportError("Please install aliyunsdkcore before initialize alinlp backend.")
    if access_key_id and access_key_secret:
        # 创建AcsClient实例
        client = AcsClient(access_key_id,access_key_secret,"cn-hangzhou")
    else:
        # please set ALINLP key & secret in your environment
        access_key_id = os.environ.get("ALINLP_AK_ENV")
        access_key_secret = os.environ.get('ALINLP_SK_ENV')
        if not access_key_id or not access_key_secret:
            raise Exception(
                "Please set ALINLP_AK_ENV and ALINLP_SK_ENV in your environment"
            )
        client = AcsClient(access_key_id,access_key_secret,"cn-hangzhou")

    request = CommonRequest()
    # domain和version是固定值
    request.set_domain(os.getenv("ALINLP_DOMAIN","alinlp.cn-hangzhou.aliyuncs.com"))
    request.set_version(os.getenv("ALINLP_VERSION","2020-06-29"))

    # action name可以在API文档里查到
    request.set_action_name('GetWeChGeneral')

    # 需要add哪些param可以在API文档里查到
    request.add_query_param('ServiceCode', 'alinlp')
    if size is None:
        size = os.getenv("EMBEDDING_DIM","100")
    request.add_query_param('Size', size)
    if api_type is not None:
        request.add_query_param('Type', api_type) # 自动分词
    if operation is not None:
        request.add_query_param('Operation', operation) # 自动分词

    # print(f"texts ================================= : {texts}")
    # Convert single text to list if needed
    if isinstance(texts, str):
        texts = [texts]
    
    len1 = len(texts)
    # 去掉空字符串或去掉空格后为空的元素
    texts = [t for t in texts if t and t.strip()]
    len2 = len(texts)
    if len1 != len2:
        print(f"Removed {len1 - len2} empty texts.")

    print(f"Embedding start ===========================")
    embeddings = []
    for text in texts:
        try:
            # print(f"Embedding text ================= : {text}")
            request.add_query_param('Text', text)

            # if text is None or text.isspace() or text == "":
            #     print(f"Skipping empty text: {text}")
            #     continue
            # 打印完整的URL路径和参数
            endpoint = f"https://{request.get_domain()}/"
            params = request.get_query_params()
            # 拼接参数字符串
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            if not param_str:
                print(f"No query parameters found. text: {text}")
            full_url = f"{endpoint}?{param_str}"
            # print(f"Request URL: {full_url}")

            response = client.do_action_with_exception(request)
            resp_obj = json.loads(response)
            data = json.loads(resp_obj["Data"])  # 先解析 Data 字符串
            vec = data["result"]["vec"]          # 再取 vec 数组
            # print(f"vector length: {len(vec)}")
            embeddings.append(vec)            
        except Exception as e:
            logger.error(f"Error calling ALINLP Embedding API: {str(e)} - {full_url}")
            print(f"Error calling ALINLP Embedding API: {str(e)} - {full_url}")
            # 若API返回异常，填充一个全0的向量，长度为size参数
            embeddings.append([0.0] * int[size])
            #break

    print(f"Embedding end ===========================")
    return np.array(embeddings)
