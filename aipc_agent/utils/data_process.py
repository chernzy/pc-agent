from typing import List, Tuple
from openai.types.chat import ChatCompletionMessageParam
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper
)

from utils.protocol import Role

def parse_messages(
        messages: List[ChatCompletionMessageParam], split_role=Role.USER
) -> Tuple[str, List[List[ChatCompletionMessageParam]]]:
    system, rounds = "", []
    r = []
    for i, message in enumerate(messages):
        if message["role"] == Role.SYSTEM:
            system = message["content"]
            continue
        if message["role"] == split_role and r:
            rounds.append(r)
            r = []
        r.append(message)
    if r:
        rounds.append(r)
    return system, rounds

def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def is_partial_stop(output: str, stop_str: str):
    return any(stop_str.startswith(output[-i:]) for i in range(0, min(len(output), len(stop_str))))


SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config) -> int:
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_scaling_factor = config.rope_scaling["factor"] if rope_scaling else 1
    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048

def apply_stopping_strings(reply: str, stop_strings: List[str]) -> Tuple[str, bool]:
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
                else:
                    continue
            break

    return reply, stop_found