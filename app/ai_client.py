
from typing import Optional

import httpx
import openai
import urllib3

class OpenAIClient:
    def __init__(
            self,
            api_key: Optional[str] = None,
            verify_ssl: bool = False
        ):
        if api_key:
            openai.api_key = api_key

        if not openai.api_key:
            raise ValueError("OpenAI API key must be set")
        
        if verify_ssl:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            http = httpx.Client(verify=False)
            self.client = openai.OpenAI(
                api_key=openai.api_key,
                http_client=http
            )