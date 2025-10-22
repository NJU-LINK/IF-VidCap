from openai import OpenAI
from google import genai
from google.genai import types
import json
from pathlib import Path
import concurrent.futures
import functools
from typing import Callable, Any

def gemini_client_2077(api_config_path: str='./api.json'):
    """
    配置 Gemini 模型服务客户端
    """
    api_config = json.loads(Path(api_config_path).read_text(encoding='utf-8'))
    client = genai.Client(api_key=api_config["api_key"],
                    http_options=types.HttpOptions(base_url=api_config["gemini_url"]))
    return client

def gemini_client(api_config_path: str='./api.json'):
    """
    配置 Gemini 模型服务客户端
    """
    api_config = json.loads(Path(api_config_path).read_text(encoding='utf-8'))
    client = genai.Client(api_key=api_config["gemini_key"])
    return client

def openai_client(api_config_path: str='./api.json'):
    """
    配置 OpenAI 模型服务客户端
    """
    api_config = json.loads(Path(api_config_path).read_text(encoding='utf-8'))
    client = OpenAI(api_key=api_config["api_key"],
                    base_url=api_config["openai_url"],
                    timeout=18000)
    return client

# 工具内容
def clean_json_response(response_text: str) -> str:
    """
    清理响应文本，移除markdown代码块标记
    """
    if response_text.startswith("```json\n"):
        response_text = response_text[8:]
    if response_text.endswith("\n```"):
        response_text = response_text[:-4]
    return response_text


def timeout_with_retry(timeout_seconds: int, max_retries: int = 3):
    """使用线程池的超时装饰器，支持重试"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        try:
                            result = future.result(timeout=timeout_seconds)
                            return result
                        except concurrent.futures.TimeoutError:
                            future.cancel()  # 尝试取消任务
                            raise TimeoutError(f"函数 {func.__name__} 执行超时 ({timeout_seconds}秒)")
                            
                except TimeoutError as e:
                    print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
                    if attempt == max_retries - 1:
                        raise
                    print("正在重试...")
                    
            return None
        return wrapper
    return decorator


def error_retry(max_retries: int = 3, exceptions: tuple = (Exception,), delay: float = 1.0, backoff: float = 2.0):
    """
    出错重试装饰器
    
    Args:
        max_retries: 最大重试次数
        exceptions: 需要捕获并重试的异常类型元组
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的倍数递增因子
    """
    import time
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            
            for attempt in range(max_retries + 1):  # +1 因为第一次不算重试
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        print(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败")
                        raise
                    
                    print(f"函数 {func.__name__} 第 {attempt + 1} 次执行失败: {type(e).__name__}: {e}")
                    print(f"等待 {current_delay:.1f} 秒后进行第 {attempt + 2} 次尝试...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff  # 指数退避
                    
            return None
        return wrapper
    return decorator


def combined_retry(timeout_seconds: int = 30, timeout_retries: int = 3, 
                  error_retries: int = 3, exceptions: tuple = (Exception,), 
                  delay: float = 1.0, backoff: float = 2.0):
    """
    组合装饰器：同时支持超时重试和出错重试
    
    Args:
        timeout_seconds: 超时时间
        timeout_retries: 超时重试次数
        error_retries: 出错重试次数
        exceptions: 需要捕获的异常类型
        delay: 初始延迟时间
        backoff: 延迟倍数
    """
    def decorator(func: Callable) -> Callable:
        # 先应用出错重试，再应用超时重试
        func_with_error_retry = error_retry(error_retries, exceptions, delay, backoff)(func)
        func_with_both_retries = timeout_with_retry(timeout_seconds, timeout_retries)(func_with_error_retry)
        return func_with_both_retries
    return decorator