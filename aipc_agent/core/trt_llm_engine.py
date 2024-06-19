import os
import traceback
from transformers import AutoTokenizer
from schemas.openai_completion_params import ChatCompletionMessageParam
from typing import (
    Optional,
    Any,
    Dict,
    Callable,
    Tuple,
    Union,
    List,
    Iterator
)

from schemas.openai_schema import (
    ChatCompletionMessage,
    ChatCompletion,
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    FunctionCall,
    ChatCompletionMessageToolCall,
    Completion,
    CompletionChoice, 
    Logprobs,
    CompletionUsage
)
from schemas.openai_chunk_schema import Choice as ChunkChoice
from schemas.openai_chunk_schema import ChatCompletionChunk

try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        PrivateAttr,
        root_validator,
        validator,
        create_model,
        StrictFloat,
        StrictInt,
        StrictStr,
    )
    from pydantic.v1.fields import FieldInfo
    from pydantic.v1.error_wrappers import ValidationError
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        PrivateAttr,
        root_validator,
        validator,
        create_model,
        StrictFloat,
        StrictInt,
        StrictStr,
    )
    from pydantic.fields import FieldInfo
    from pydantic.error_wrappers import ValidationError
import gc
import json
import torch
import numpy as np
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner, ModelRunnerCpp
from tensorrt_llm.logger import logger
import tensorrt_llm
from pathlib import Path
import uuid
import time
from fastapi.responses import JSONResponse

from utils.template import get_prompt_adapter
from utils.trt_utils import read_model_name, load_tokenizer, throttle_generator
from utils.constants import DEFAULT_HF_MODEL_DIRS, DEFAULT_NUM_OUTPUTS, DEFAULT_CONTEXT_WINDOW, EOS_TOKEN, PAD_TOKEN, EOS
from utils.constants import ErrorCode
from utils.request import create_error_response

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)

def make_resData(data, chat=False, promptToken=[]):
    resData = {
        "id": f"chatcmpl-{str(uuid.uuid4())}" if (chat) else f"cmpl-{str(uuid.uuid4())}",
        "object": "chat.completion" if (chat) else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA",
        "usage": {
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "total_tokens": data["prompt_tokens"] + data["completion_tokens"]
        }
    }
    if (len(promptToken) != 0):
        resData["promptToken"] = promptToken
    if (chat):
        # only one choice is supported
        resData["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"],
            },
            "finish_reason": "stop" if data["stopped"] else "length"
        }]
    else:
        # only one choice is supported
        resData["choices"] = [{
            "text": data["content"],
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if data["stopped"] else "length"
        }]
    return resData


def make_resData_stream(data, chat=False, start=False):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion.chunk" if (chat) else "text_completion.chunk",
        "created": int(time.time()),
        "model": "LLaMA",
        "choices": [
            {
                "finish_reason": None,
                "index": 0
            }
        ]
    }
    slot_id = data["slot_id"]
    if (chat):
        if (start):
            resData["choices"][0]["delta"] = {
                "role": "assistant"
            }
        else:
            resData["choices"][0]["delta"] = {
                "content": data["content"]
            }
            if (data["stop"]):
                resData["choices"][0]["finish_reason"] = "stop" if data["stopped"]  else "length"
    else:
        resData["choices"][0]["text"] = data["content"]
        if (data["stop"]):
            resData["choices"][0]["finish_reason"] = "stop" if data["stopped"] else "length"

    return resData

class TrtLLMEngine(BaseModel):

    model_path: Optional[str] = Field(
        description="The path to the trt engine."
    )
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(description="Whether to print verbose output.")

    _model: Any = PrivateAttr()
    _model_name = PrivateAttr()
    _model_version = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _pad_id:Any = PrivateAttr()
    _end_id: Any = PrivateAttr()
    _max_new_tokens = PrivateAttr()
    _max_input_tokens = PrivateAttr()
    _sampling_config = PrivateAttr()
    _debug_mode = PrivateAttr()
    _add_special_tokens = PrivateAttr()
    _verbose = PrivateAttr()
    _prompt_name = PrivateAttr()
    _prompt_adapter = PrivateAttr()
    _construct_prompt = PrivateAttr()

    def __init__(
            self,
            model_path: Optional[str] = None,
            engine_name: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            vocab_file: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            prompt_name: Optional[str] = None,
            prompt_template = None,
            generate_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            use_py_session = True,
            add_special_tokens = False,
            trtLlm_debug_mode = False,
            verbose: bool = False
    ) -> None:
        runtime_rank = tensorrt_llm.mpi_rank()
        self._model_name, self._model_version = read_model_name(model_path)
        if tokenizer_dir is None:
            logger.warning(
                "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
            )

            if self._model_name == "GemmaForCausalLM":
                tokenizer_dir = 'gpt2'
            else:
                tokenizer_dir = DEFAULT_HF_MODEL_DIRS[self._model_name]

        self._max_input_tokens=context_window
        self._add_special_tokens=add_special_tokens
        self._verbose = verbose
        self._prompt_name = None
        self._prompt_adapter = get_prompt_adapter(self._model_name, prompt_name=self._prompt_name)
        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})
        #logger.set_level('verbose')
        self._tokenizer, self._pad_id, self._end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=vocab_file,
            model_name=self._model_name,
            model_version=self._model_version,
            #tokenizer_type=args.tokenizer_type,
        )

        runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
        if verbose:
            print(f"[ChatRTX] Trt-llm mode debug mode: {trtLlm_debug_mode}")

        runner_kwargs = dict(engine_dir=model_path,
                             rank=runtime_rank,
                             debug_mode=trtLlm_debug_mode,
                             lora_ckpt_source='hf')

        if not use_py_session:
            runner_kwargs.update(free_gpu_memory_fraction = 0.5)

        self._model = runner_cls.from_dir(**runner_kwargs)
        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
           {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        self._max_new_tokens = max_new_tokens
        self._check_construct_prompt()

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            prompt_name=prompt_name,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TrtLlmAPI"
    
    def _check_construct_prompt(self) -> None:
        """ Check whether to need to construct prompts or inputs. """
        self._construct_prompt = self._prompt_name is not None
        if "chatglm3" in self._model_name:
            logger.info("Using ChatGLM3 Model for Chat!")
        else:
            self._construct_prompt = True
    
    
    def convert_to_inputs(
            self,
            prompt_or_messages: Union[List[ChatCompletionMessageParam], str],
            infilling: Optional[bool] = False,
            suffix_first: Optional[bool] = False,
            **kwargs
    ) -> Tuple[Union[List[int], Dict[str, Any]], Union[List[ChatCompletionMessageParam], str]]:
        """
        Convert the prompt or messages into input format for the model.

        Args:
            prompt_or_messages: The prompt or messages to be converted.
            infilling: Whether to perform infilling.
            suffix_first: Whether to append the suffix first.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing the converted inputs and the prompt or messages.
        """
        if isinstance(prompt_or_messages, str):
            raise NotImplementedError("String input is not supported.")
        else:
            inputs, prompt_or_messages = self.apply_chat_template(prompt_or_messages, **kwargs)

        return inputs, prompt_or_messages
    
    def apply_chat_template(
            self,
            messages: List[ChatCompletionMessageParam],
            max_new_tokens: Optional[int] = 256,
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            **kwargs
    ) -> Tuple[Union[List[int], Dict[str, Any]], Optional[str]]:
        """
        Apply chat template to generate model inputs and prompt.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
            max_new_tokens (Optional[int], optional): Maximum number of new tokens to generate. Defaults to 256.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Union[List[int], Dict[str, Any]], Union[str, None]]: Tuple containing the generated inputs and prompt.
        """
        if self._prompt_adapter.function_call_available:
            messages = self._prompt_adapter.postprocess_messages(messages, functions, tools=tools)
            if functions or tools:
                logger.debug(f" =========== messages with tools ============= \n {messages}")
        
        if self._construct_prompt:
            if getattr(self._tokenizer, "chat_template", None) and not self._prompt_name:
                prompt = self._tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self._prompt_adapter.apply_chat_template(messages)
        return prompt, None
    
    def parse_input(self,
                    tokenizer,
                    input_text=None,
                    prompt_template=None,
                    input_file=None,
                    add_special_tokens=False,
                    max_input_length=4096,
                    pad_id=None,
                    num_prepend_vtokens=[],
                    model_name=None,
                    model_version=None):
        if pad_id is None:
            pad_id = tokenizer.pad_token_id
        if model_name == "GemmaForCausalLM":
            add_special_tokens = True
        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(curr_text)
                input_ids = tokenizer.encode(curr_text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_input_length)
                batch_input_ids.append(input_ids)
        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = tokenizer.vocab_size - len(tokenizer.special_tokens_map.get("additional_special_tokens", []))
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = list(range(base_vocab_size, base_vocab_size + length)) + batch_input_ids[i]
        
        if model_name == "ChatGLMForCausalLM" and model_version == "glm":
            for ids in batch_input_ids:
                ids.append(tokenizer.sop_token_id)

        batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
        return batch_input_ids
    
    def print_output(
            self,
            tokenizer,
            output_ids,
            input_lengths,
            sequence_lengths,
            output_csv=None,
            output_npy=None,
            context_logits=None,
            generation_logits=None,
            output_logits_npy=None
    ):
        output_text = ""
        batch_size, num_beams, _ = output_ids.size()
        if output_csv is None and output_npy is None:
            for batch_idx in range(batch_size):
                inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist()
                for beam in range(num_beams):
                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                    output_text = tokenizer.decode(outputs)
        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        return output_text, output_ids
    
    def complete(self, params: Dict[str, Any]) -> Any:
        self.generate_kwargs.update({"stream": False})
        is_formatted = params.pop("formatted", False)
        if not is_formatted:
            prompt_or_messages = params.get("prompt_or_messages")
            inputs, prompt = self.convert_to_inputs(
                prompt_or_messages, 
                infilling=params.get("infilling", False), 
                suffix_first=params.get("suffix_first", False),
                max_new_tokens=params.get("max_tokens", 256),
                functions=params.get("functions"),
                tools=params.get("tools")
            )
        if self._verbose:
            print(f"[AIPC Service] Context send to LLM \n: {inputs}")
        input_text = [inputs]
        batch_input_ids = self.parse_input(
            tokenizer=self._tokenizer,
            input_text=input_text,
            prompt_template=None,
            input_file=None,
            add_special_tokens=self._add_special_tokens,
            max_input_length=self._max_input_tokens,
            pad_id=self._pad_id,
            num_prepend_vtokens=None,
            model_name=self._model_name,
            model_version=self._model_version
        )
        input_lengths = [x.size(0) for x in batch_input_ids]

        if self._verbose:
            print(f"[AIPC Service] Number of token: {input_lengths[0]}")

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    batch_input_ids,
                    max_new_tokens=self._max_new_tokens,
                    max_attention_window_size=4096,
                    end_id=self._end_id,
                    pad_id=self._pad_id,
                    temperature=1.0,
                    top_k=1,
                    top_p=0,
                    num_beams=1,
                    length_penalty=1.0,
                    early_stopping=False,
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    stop_words_list=None,
                    bad_words_list=None,
                    lora_uids=None,
                    prompt_table_path=None,
                    prompt_tasks=None,
                    streaming=False,
                    output_sequence_lengths=True,
                    return_dict=True
                )
                torch.cuda.synchronize()
            outputs["error_code"] = 0
            output_ids = outputs["output_ids"]
            sequence_lengths = outputs["sequence_lengths"]
            output_text, output_token_ids = self.print_output(self._tokenizer, output_ids, input_lengths, sequence_lengths)
            torch.cuda.empty_cache()
            gc.collect()
            yield {
                "text": output_text,
                "output_token_ids": output_token_ids
            }
        except torch.cuda.OutOfMemoryError as e:
            yield {
                "text": f"{server_error_msg}\n\n{e}",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY
            }
        except (ValueError, RuntimeError) as e:
            traceback.print_exc()
            yield {
                "text": f"{server_error_msg}\n\n{e}",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }


    def stream_complete(self, params: Dict[str, Any]) -> Any:
        is_formatted = params.pop("formatted", False)
        if not is_formatted:
            prompt_or_messages = params.get("prompt_or_messages")
            inputs, prompt = self.convert_to_inputs(
                prompt_or_messages, 
                infilling=params.get("infilling", False), 
                suffix_first=params.get("suffix_first", False),
                max_new_tokens=params.get("max_tokens", 256),
                functions=params.get("functions"),
                tools=params.get("tools")
            )
        if self._verbose:
            print(prompt)
        input_text = [inputs]
        batch_input_ids = self.parse_input(
            tokenizer=self._tokenizer,
            input_text=input_text,
            prompt_template=None,
            input_file=None,
            add_special_tokens=self._add_special_tokens,
            max_input_length=self._max_input_tokens,
            pad_id=self._pad_id,
            num_prepend_vtokens=None,
            model_name=self._model_name,
            model_version=self._model_version
        )
        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = self._model.generate(
                batch_input_ids,
                max_new_tokens=self._max_new_tokens,
                max_attention_window_size=4096,
                sink_token_length=None,
                end_id=self._end_id,
                pad_id=self._pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=True,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop_words_list=None,
                bad_words_list=None,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=True,
                output_sequence_lengths=True,
                return_dict=True
            )
            torch.cuda.synchronize()
        previous_text = ""
        def gen():
            nonlocal previous_text
            for curr_outputs in throttle_generator(outputs, 5):
                output_ids = curr_outputs["output_ids"]
                sequence_lengths = curr_outputs["sequence_lengths"]
                output_text, output_token_ids = self.print_output(self._tokenizer, output_ids, input_lengths, sequence_lengths)

                torch.cuda.synchronize()
                if output_text.endswith("</s>"):
                    output_text = output_text[:-4]
                pre_token_len = len(previous_text)
                new_text = output_text[pre_token_len:]
                yield new_text
                previous_text = output_text
        return gen()

    def chat(self, params: Dict[str, Any]) -> Any:
        outputs = self.complete(params)
        for output in outputs:
            message = ChatCompletionMessage(
                role="assistant",
                content=output["text"],
            )
        choice = Choice(
            index = 0,
            message=message,
            finish_reason="stop",
            logprobs=None
        )
        return ChatCompletion(
            id=f"chat{str(uuid.uuid4())}",
            choices=[choice],
            created=int(time.time()),
            model=self._model_name,
            object="chat.completion",
            usage=None
        )


    def stream_chat(self, params: Dict[str, Any]) -> Any:
        """
        Creates a chat completion stream.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.

        Yields:
            Dict[str, Any]: The output of the chat completion stream.
        """
        outputs = self.stream_complete(params)
        for i in outputs:
            if i == "":
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="")
                )
                yield ChatCompletionChunk(
                    id=f"chat{str(uuid.uuid4())}",
                    choices=[choice],
                    created=int(time.time()),
                    model=self._model_name,
                    object="chat.completion.chunk",
                    system_fingerprint="fp-baichuan2-7b-chat"
                )
            finish_reason = None
            delta = ChoiceDelta(content=i, role="assistant")
            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
                logprobs=None
            )

            yield ChatCompletionChunk(
                id=f"chat{str(uuid.uuid4())}",
                choices=[choice],
                created=int(time.time()),
                model=self._model_name,
                object="chat.completion.chunk",
                system_fingerprint="fp-baichuan2-7b-chat"
            )
        finish_reason = "stop"
        delta = ChoiceDelta(content="", role="assistant")
        choice = ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
            logprobs=None
        )
        yield ChatCompletionChunk(
            id=f"chat{str(uuid.uuid4())}",
            choices=[choice],
            created=int(time.time()),
            model=self._model_name,
            object="chat.completion.chunk",
            system_fingerprint="fp-baichuan2-7b-chat"
        )
        
    
    def remove_extra_eos_ids(self, outputs):
        outputs.reverse()
        while outputs and outputs[0] == 2:
            outputs.pop(0)
        outputs.reverse()
        outputs.append(2)
        return outputs
    
    def get_output(self, output_ids, input_lengths, max_output_len, tokenizer):
        num_beams = output_ids.size(1)
        output_text = ""
        outputs = None
        for b in range(input_lengths.size(0)):
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                outputs = self.remove_extra_eos_ids(outputs)
                output_text = tokenizer.decode(outputs)
        return output_text, outputs
    
    def generate_completion_dict(self, text_str):
        """
        Generate a dictionary for text completion details.
        Returns:
        dict: A dictionary containing completion details.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = self._model if self._model is not None else self.model_path
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": 'stop'
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }
    
    def create_chat_completion(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        params = params or {}
        params.update(kwargs)
        return (
            self.stream_chat(params)
            if params.get("stream", False)
            else self.chat(params)
        )

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return {
            "strings": ["<reserved_106>", "<reserved_107>"],
            "token_ids": [195, 196],
        }
    
    def unload_model(self):
        if self._model is not None:
            del self._model
        torch.cuda.empty_cache()
        gc.collect()