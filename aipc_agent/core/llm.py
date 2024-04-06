from langchain_openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
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
    tokenizer = AutoTokenizer.from_pretrained("D:\\Ubuntu\\code\\models\\chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("D:\\Ubuntu\\code\\models\\chatglm3-6b", trust_remote_code=True).to(device).eval()
    return tokenizer, model
