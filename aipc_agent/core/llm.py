from langchain_openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import AutoPeftModelForCausalLM
import torch

device = "cuda:0"

def llm_openai():
    llm = OpenAI()
    return llm

def llm_gemma2b_local():
    tokenizer = AutoTokenizer.from_pretrained("D:\Code\models\gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("D:\Code\models\gemma-2b-it", torch_dtype=torch.bfloat16).to(device)
    return tokenizer, model

def llm_glm3_6b_local():
    tokenizer = AutoTokenizer.from_pretrained("D:\\Code\\models\\chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("D:\\Code\\models\\chatglm3-6b", trust_remote_code=True, load_in_4bit=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return tokenizer, model

def llm_baichuan2_7B_local():
    tokenizer = AutoTokenizer.from_pretrained("D:\Code\models\Baichuan2-7B-Chat", trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained("D:\Code\models\hanbin_baichuan2-7B-chat-lora",load_in_4bit=True,trust_remote_code=True, low_cpu_mem_usage=True)
    return tokenizer, model