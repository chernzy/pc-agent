import gc
import time
import uuid
from threading import Thread
from types import MethodType
from typing import (
    Iterable,
    Dict,
    Any,
)
import torch
from transformers import (
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils.data_process import (
    prepare_logits_processor,
    is_partial_stop,
    apply_stopping_strings
)

@torch.inference_mode()
def generate_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any]
):
    input_ids = params.get("inputs")
    prompt = params.get("prompt")
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))
    max_new_tokens = int(params.get("max_tokens", 256))
    logprobs = params.get("logprobs")
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop")

    stop_token_ids = params.get("stop_token_ids") or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(temperature, repetition_penalty, top_p, top_k)

    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    device = model.device
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(input_ids=torch.as_tensor([input_ids], device=device))
        start_ids = torch.as_tensor([[model.generation_config.decoder_start_token_id]], dtype=torch.int64, device=device)
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values, sent_interrupt = None, False
    token_logprobs = [None]
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    previous_text = ""
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=start_ids, encoder_hidden_states=encoder_output, use_cache=True,)
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(shift_input_ids[0].tolist(), shift_logits[0]):
                    token_logprobs.append(logit[label_id])

        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=torch.as_tensor([output_ids if sent_interrupt else [token]], device=device),
                                    encoder_hidden_states=encoder_output,
                                    use_cache=True,
                                    past_key_values=None if sent_interrupt else past_key_values,)
                sent_interrupt = False
                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # switch to cpu by avoiding some bugs in mps backend
            last_token_logits = last_token_logits.float().to("cpu")
        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]

        token = tokens[0]
        output_ids.append(token)

        if logprobs is not None:
            token_logprobs.append(torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist())

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # yield the output tokens
        if i % 2 == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len(prompt)
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0
            output = tokenizer.decode(
                tmp_output_ids,
            )