from enum import IntEnum

CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 30
WORKER_API_TIMEOUT = 20


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006

DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPTForCausalLM': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
}

EOS_TOKEN = 2
PAD_TOKEN = 2

DEFAULT_CONTEXT_WINDOW = 3900
DEFAULT_NUM_OUTPUTS = 256
BOS, EOS = "<s>", "</s>"