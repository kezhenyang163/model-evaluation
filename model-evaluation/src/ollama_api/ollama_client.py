import requests
import json
from typing import Dict, List, Optional, Union, Iterator, Any
import time


class ChatCompletions:
    def __init__(self, client):
        self.client = client

    def create(self,
               model: str,
               messages: List[Dict[str, str]],
               temperature: Optional[float] = None,
               top_p: Optional[float] = None,
               max_tokens: Optional[int] = None,
               stream: bool = False,
               **kwargs) -> Union[Dict, Iterator]:
        """
        模仿OpenAI的chat.completions.create方法

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度值
            top_p: 核采样概率
            max_tokens: 最大token数
            stream: 是否流式响应
            **kwargs: 其他参数

        Returns:
            响应对象
        """
        return self.client._chat(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )


class Chat:
    def __init__(self, client):
        self.completions = ChatCompletions(client)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", api_key: Optional[str] = None):
        """
        初始化Ollama客户端

        Args:
            base_url: Ollama服务地址
            api_key: API密钥（保留参数，Ollama通常不需要）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
        }

        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'

        # 创建chat属性以模仿OpenAI接口
        self.chat = Chat(self)

    def _chat(self,
              model: str,
              messages: List[Dict[str, str]],
              temperature: Optional[float] = None,
              top_p: Optional[float] = None,
              max_tokens: Optional[int] = None,
              stream: bool = False,
              **kwargs) -> Union[Dict, Iterator]:
        """
        调用Ollama聊天API（内部方法，通过chat.completions.create调用）

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度值
            top_p: 核采样概率
            max_tokens: 最大token数
            stream: 是否流式响应
            **kwargs: 其他参数

        Returns:
            响应字典或迭代器
        """
        url = f"{self.base_url}/api/chat"

        # 构建请求数据
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # 添加可选参数
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # 添加其他可选参数到options中
        for key, value in kwargs.items():
            if key not in data:  # 避免覆盖已有参数
                options[key] = value

        if options:
            data["options"] = options

        # 发送请求
        response = requests.post(url, json=data, headers=self.headers, stream=stream)
        response.raise_for_status()

        if stream:
            return self._handle_stream_response(response)
        else:
            return self._format_response(response.json())

    def _handle_stream_response(self, response: requests.Response) -> Iterator[Dict]:
        """
        处理流式响应，并格式化为类似OpenAI的格式
        """
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.strip():
                    chunk_data = json.loads(line)
                    # 格式化为OpenAI风格的流式响应
                    formatted_chunk = {
                        "id": f"chatcmpl-{hash(str(chunk_data))}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "ollama-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk_data.get('message', {}).get('content', '')
                                    if 'message' in chunk_data else ''
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    yield formatted_chunk

    def _format_response(self, response_data: Dict) -> Dict:
        """
        将Ollama响应格式化为OpenAI风格
        """
        return {
            "id": f"chatcmpl-{hash(str(response_data))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_data.get('model', 'ollama-model'),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data.get('message', {}).get('content', '')
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    def generate(self,
                 model: str,
                 prompt: str,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 stream: bool = False,
                 **kwargs) -> Union[Dict, Iterator]:
        """
        调用Ollama生成API
        """
        url = f"{self.base_url}/api/generate"

        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        for key, value in kwargs.items():
            if key not in data:
                options[key] = value

        if options:
            data["options"] = options

        response = requests.post(url, json=data, headers=self.headers, stream=stream)
        response.raise_for_status()

        if stream:
            return self._handle_generate_stream_response(response)
        else:
            return response.json()

    def _handle_generate_stream_response(self, response: requests.Response) -> Iterator[Dict]:
        """
        处理generate API的流式响应
        """
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.strip():
                    yield json.loads(line)

    def list_models(self) -> Dict:
        """
        列出可用的模型
        """
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def show_model(self, model: str) -> Dict:
        """
        显示模型详细信息
        """
        url = f"{self.base_url}/api/show"
        data = {"name": model}
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()


# 为了完全兼容OpenAI的调用方式，创建这个类
class OpenAI:
    def __init__(self, base_url: str = "http://localhost:11434", api_key: Optional[str] = None):
        """
        完全模仿OpenAI客户端的初始化方式
        """
        self.base_url = base_url
        self.api_key = api_key
        self.client = OllamaClient(base_url, api_key)

    @property
    def chat(self):
        return self.client.chat


# 使用示例
if __name__ == "__main__":
    # 方式1：使用完全兼容OpenAI的方式
    # client = OllamaClient(base_url="http://localhost:11434")
    # client = OpenAI(base_url="http://localhost:11434")

    # 使用chat.completions.create()方式调用
    # response = client.chat.completions.create(
    #     model="deepseek-r1",
    #     messages=[
    #         {"role": "user", "content": "请介绍一下你自己"}
    #     ],
    #     temperature=0.7,
    #     top_p=0.9,
    #     max_tokens=1000
    # )
    #
    # print("Response:", response)
    # print("Assistant reply:", response["choices"][0]["message"]["content"])



    # 流式响应
    # print("\nStreaming response:")
    # stream_response = client.chat.completions.create(
    #     model="deepseek-r1",
    #     messages=[
    #         {"role": "user", "content": "给我讲个小故事"}
    #     ],
    #     stream=True
    # )
    #
    # for chunk in stream_response:
    #     if chunk["choices"][0]["delta"].get("content"):
    #         print(chunk["choices"][0]["delta"]["content"], end='', flush=True)



    # # 方式2：直接使用OllamaClient（也支持chat.completions.create）
    ollama_client = OllamaClient(base_url="http://localhost:11434")
    response = ollama_client.chat.completions.create(
        model="deepseek-r1",
        messages=[{"role": "user", "content": "给我讲一个小故事 100字以内"}],
        temperature=0.5,
        max_tokens=1000
    )
    print(f"\nDirect client response: {response['choices'][0]['message']['content']}")