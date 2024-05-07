from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy

# from openai.types.chat import ChatCompletionMessageParam
from schemas.openai_completion_params import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer

from utils.data_process import parse_messages
from utils.protocol import Role


TOOL_DESC = """
> Tool Name: {tool_name}
Tool Description: {tool_description}

"""

TOOL_ARGS = """
- {arg_name}: ({arg_type})
{arg_description}
"""

REACT_INSTRUCTION = """
You are a helpful assistant with access to the following functions. Use them if required.

Edge cases you must handle:
 - If there are no functions that match the user request, you will respond politely that you cannot help. 
You have access to the following tools:
{tools_text}
"""


def build_baichuan_chat_input(
        tokenizer: PreTrainedTokenizer,
        messages: List[ChatCompletionMessageParam],
        context_len: int = 4096,
        max_new_tokens: int = 256
) -> List[int]:
    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if message["role"] == Role.USER:
                round_tokens.append(195)
            else:
                round_tokens.append(196)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != Role.ASSISTANT:
        input_tokens.append(196)

    return input_tokens[-max_input_tokens:]

def check_is_baichuan(model) -> bool:
    return "BaichuanLayer" in getattr(model, "_no_split_modules", [])