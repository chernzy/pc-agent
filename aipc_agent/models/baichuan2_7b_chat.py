from abc import ABC, abstractclassmethod
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from schemas.openai_schema import ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionResponse, FunctionCallResponse
from utils.data_process import process_baichuan_messages

class Baichuan2_7B_Chat(ABC):
    
    def init_model(self):
        self.model = AutoPeftModelForCausalLM("D:\Code\models\hanbin_baichuan2-7B-chat-lora",load_in_4bit=True,trust_remote_code=True, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained("D:\Code\models\Baichuan2-7B-Chat", trust_remote_code=True)
        return self.model, self.tokenizer
    
    async def predict(self, model, tokenizer, params: dict):
        choice_data = ChatCompletionResponseStreamChoice(
            index = 0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model="Baichuan2-7B-Chat", choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))



    async def generate_stream(self, model, tokenizer, params: dict):
        messages = params["messages"]
        tools = params["tools"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_tokens", 256))
        messages = process_baichuan_messages(messages, tools = tools)
        query, role = messages[-1]["content"], messages[-1]["role"]
        