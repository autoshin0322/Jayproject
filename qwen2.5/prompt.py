# docker cp <로컬경로> <컨테이너ID 또는 이름>:<컨테이너 내부 경로>
from dataclasses import dataclass
from typing import List

@dataclass
class PromptMessage:
    role: str
    content: str

@dataclass
class LLMPrompt:
    messages: List[PromptMessage]
    ref: str = None

@dataclass
class LLMResult:
    meta: str
    prompt_ref: str
    message_ref: str