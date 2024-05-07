import traceback
import json
from abc import ABC
from typing import (
    Optional,
    List,
    Union,
    Tuple,
    Dict,
    Iterator,
    Any,
)

import torch
from fastapi.responses import JSONResponse
from loguru import logger
# from openai.types.chat import (
#     # ChatCompletionMessage,
#     # ChatCompletion,
#     # ChatCompletionChunk,
#     ChatCompletionMessageParam
# )

from schemas.openai_completion_params import ChatCompletionMessageParam

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

# from openai.types.chat.chat_completion import Choice
# from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
# from openai.types.chat.chat_completion_chunk import (
#     ChoiceDelta,
#     ChoiceDeltaFunctionCall,
#     ChoiceDeltaToolCall
# )
# from openai.types.chat.chat_completion_message import FunctionCall
# from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
# from openai.types.completion import Completion
# from openai.types.completion_choice import CompletionChoice, Logprobs
# from openai.types.completion_usage import CompletionUsage
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.template import get_prompt_adapter
from utils.generations import (
    build_baichuan_chat_input,
    check_is_baichuan,
    generate_stream_v2,
    generate_stream
)
from utils.data_process import get_context_length
from utils.compat import model_validate
from utils.constants import ErrorCode
from utils.request import create_error_response

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC, PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)

class DefaultEngine(ABC):
    """
    transformer model engine
    """
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            model_name: str,
            context_len: Optional[int] = None,
            prompt_name: Optional[str] = None,
            use_streamer_v2: Optional[bool] = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.context_len = context_len
        self.use_streamer_v2 = use_streamer_v2
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)
        
        self._prepare_for_generate()
        self._patch_tokenizer()

    def _prepare_for_generate(self) -> None:
        self.generate_stream_func = generate_stream
        # TODO: add multiple model support
        self._check_construct_prompt()

        if self.context_len is None:
            self.context_len = get_context_length(self.model.config)

    
    def _check_construct_prompt(self) -> None:
        self.construct_prompt = self.prompt_name is not None
        if check_is_baichuan(self.model):
            logger.info("Using baichuan model for chat")
        else:
            self.construct_prompt = True


    def _patch_tokenizer(self) -> None:
        from utils.patcher import patch_tokenizer

        patch_tokenizer(self.tokenizer)


    def convert_to_inputs(
            self,
            prompt_or_messages: Union[List[ChatCompletionMessageParam], str],
            infilling: Optional[bool] = False,
            suffix_first: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[Union[List[int], Dict[str, Any]], Union[List[ChatCompletionMessageParam], str]]:
        if isinstance(prompt_or_messages, str):
            # TODO: add multiple models support
            if infilling:
                inputs = self.tokenizer(prompt_or_messages, suffix_first=suffix_first,).input_ids
            else:
                inputs = self.tokenizer(prompt_or_messages).input_ids
            if isinstance(inputs, list):
                max_src_len = self.context_len - kwargs.get("max_tokens", 256) - 1
                inputs = inputs[-max_src_len:]

        else:
            inputs, prompt_or_messages = self.apply_chat_template(prompt_or_messages, **kwargs)
        return inputs, prompt_or_messages
    

    def apply_chat_template(
            self,
            messages: List[ChatCompletionMessageParam],
            max_new_tokens: Optional[int] = 256,
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            **kwargs,
    ) -> Tuple[Union[List[int], Dict[str, Any]], Optional[str]]:
        if self.prompt_adapter.function_call_available and kwargs.get("tool_choice") == "auto":
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools=tools
            )
            if functions or tools:
                logger.debug(f" Messages with tools \n {messages}")

        if self.construct_prompt:
            if getattr(self.tokenizer, "chat_template", None) and not self.prompt_name:
                prompt = self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True,)
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)
            inputs = self.tokenizer(prompt).input_ids
            if isinstance(inputs, list):
                max_src_len = self.context_len - max_new_tokens - 1
                inputs = inputs[-max_src_len:]
            return inputs, prompt
        else:
            inputs = self.build_chat_inputs(
                messages, max_new_tokens, functions, tools, **kwargs
            )
        return inputs, None
    

    def build_chat_inputs(
            self,
            messages: List[ChatCompletionMessageParam],
            max_new_tokens: Optional[int] = 256,
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
    ) -> List[int]:
        if check_is_baichuan(self.model):
            inputs = build_baichuan_chat_input(self.tokenizer, messages, self.context_len, max_new_tokens)
        else:
            raise NotImplementedError

        return inputs
    

    def _generate(self, params: Dict[str, Any]) -> Iterator[dict]:
        prompt_or_messages = params.get("prompt_or_messages")
        inputs, prompt = self.convert_to_inputs(
            prompt_or_messages,
            infilling=params.get("infilling", False),
            suffix_first=params.get("suffix_first", False),
            max_new_tokens=params.get("max_tokens", 256),
            functions=params.get("functions"),
            tools=params.get("tools"),
            tool_choice=params.get("tool_choice"),
        )
        params.update(dict(inputs=inputs, prompt=prompt))

        try:
            print(params, " ------------ params ---------")
            for output in self.generate_stream_func(self.model, self.tokenizer, params):
                output["error_code"] = 0
                yield output
        except torch.cuda.OutOfMemoryError as e:
            yield {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            traceback.print_exc()
            yield {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }


    def _create_completion_stream(self, params: Dict[str, Any]) -> Iterator[Completion]:
        for output in self._generate(params):
            if output["error_code"] != 0:
                yield output
                return
            
            logprobs = None
            if params.get("logprobs") and output["logprobs"]:
                logprobs = model_validate(Logprobs, output["logprobs"])

            choice = CompletionChoice(
                index=0,
                text=output["delta"],
                finish_reason="stop",
                logprobs=logprobs,
            )
            yield Completion(
                id=output["id"],
                choices=[choice],
                created=output["created"],
                model=output["model"],
                object="text_completion",
                system_fingerprint="fp_baichuan2-7b-fc"
            )

    
    def _create_completion(self, params: Dict[str, Any]) -> Union[Completion, JSONResponse]:
        last_output = None
        for output in self._generate(params):
            last_output = output
        if last_output["error_code"] != 0:
            return create_error_response(last_output["error_code"], last_output["text"])
        
        logprobs = None
        if params.get("logprobs") and last_output["logprobs"]:
            logprobs = model_validate(Logprobs, last_output["logprobs"])

        choice = CompletionChoice(
            index=0,
            text=last_output["text"],
            finish_reason="stop",
            logprobs=logprobs,
        )
        usage = model_validate(CompletionUsage, last_output["usage"])
        return Completion(
            id=last_output["id"],
            choices=[choice],
            created=last_output["created"],
            model=last_output["model"],
            object="text_completion",
            usage=usage,
        )
    

    def _create_chat_completion_stream(self, params: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        _id, _created, _model = None, None, None
        has_function_call = False
        for i, output in enumerate(self._generate(params)):
            if output["error_code"] != 0:
                yield output
                return
            
            _id, _created, _model = output["id"], output["created"], output["model"]
            if i == 0:
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None,
                    logprobs=None,
                )
                yield ChatCompletionChunk(
                    id=f"chat{_id}",
                    choices=[choice],
                    created=_created,
                    model=_model,
                    object="chat.completion.chunk",
                    system_fingerprint="fp_baichuan2-7b-fc",
                )
            
            finish_reason = output["finish_reason"]
            if len(output["delta"]) == 0 and finish_reason != "function_call":
                continue

            function_call = None
            if finish_reason == "function_call":
                try:
                    _, function_call = self.prompt_adapter.parse_assistant_response(
                        output["text"], params.get("functions"), params.get("tools"),
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                has_function_call = True
                function_call = ChoiceDeltaFunctionCall(**function_call)
                delta = ChoiceDelta(
                    content=output["delta"],
                    function_call=function_call
                )
            elif isinstance(function_call, dict) and "function" in function_call:
                has_function_call  = True
                finish_reason = "tool_calls"
                function_call["index"] = 0
                tool_calls = [model_validate(ChoiceDeltaToolCall, function_call)]
                delta = ChoiceDelta(
                    content=output["delta"],
                )
            else:
                delta = ChoiceDelta(content=output["delta"], role="assistant")
            
            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
                # logprobs=None,
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
                system_fingerprint="fp_baichuan2-7b-fc",
            )

        if not has_function_call:
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(content="", role="assistant"),
                finish_reason="stop",
                logprobs=None
            )
            
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
                system_fingerprint="fp_baichuan2-7b-fc",
            )

    def _create_chat_completion(self, params: Dict[str, Any]) -> Union[ChatCompletion, JSONResponse]:
        last_output = None
        for output in self._generate(params):
            last_output = output
        
        if last_output["error_code"] != 0:
            return create_error_response(last_output["error_code"], last_output["text"])
        
        function_call, finish_reason = None, "stop"
        if params.get("functions") or params.get("tools"):
            try:
                res, function_call = self.prompt_adapter.parse_assistant_response(
                    last_output["text"], params.get("functions"), params.get("tools"),
                )
                last_output["text"] = res
            except Exception as e:
                traceback.print_exc()
                logger.warning("Failed to parse tool call")
        
        if isinstance(function_call, dict) and "arguments" in function_call:
            finish_reason = "function_call"
            function_call = FunctionCall(**function_call)
            message = ChatCompletionMessage(
                role="assistant",
                content=last_output["text"],
                tool_calls=function_call,
            )
        elif isinstance(function_call, dict) and "function" in function_call:
            finish_reason = "tool_calls"
            tool_calls = [model_validate(ChatCompletionMessageToolCall, function_call)]
            message = ChatCompletionMessage(
                role="assistant",
                content=last_output["text"],
                tool_calls=tool_calls,
            )
        else:
            message = ChatCompletionMessage(
                role="assistant",
                content=last_output["text"].strip(),
            )
        
        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,
            logprobs=None,
        )
        usage = model_validate(CompletionUsage, last_output["usage"])
        return ChatCompletion(
            id=f"chat{last_output['id']}",
            choices=[choice],
            created=last_output["created"],
            model=last_output["model"],
            object="chat.completion",
            usage=usage,
        )
    

    def create_completion(
            self,
            params: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Union[Iterator[Completion], Completion]:
        params = params or {}
        params.update(kwargs)
        return (
            self._create_completion_stream(params)
            if params.get("stream", False)
            else self._create_completion(params)
        )
    

    def create_chat_completion(
            self,
            params: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> Union[Iterator[ChatCompletionChunk], ChatCompletion]:
        params = params or {}
        params.update(kwargs)
        if params.get("tool_choice") == "auto":
            return self._create_chat_completion(params)
        return (
            self._create_chat_completion_stream(params)
            if params.get("stream", False)
            else self._create_chat_completion(params)
        )

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
    

