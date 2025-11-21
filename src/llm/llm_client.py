from typing import List, Dict, Any, Union

import openai
from openai.types.chat import ChatCompletion
from openai import Stream

class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str = 'fake-key'):
        self.client = openai.OpenAI(
                            base_url=base_url,
                            api_key=api_key
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Union[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream]:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        return response

