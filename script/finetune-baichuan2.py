import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from transformers import Trainer
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_traning, get_peft_model, TaskType
from datasets import load_dataset

model_name = "Baichuan2-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)

chat_template = \
"{% for message in messages %}" \
    "{% if message['from'] == 'system' %}" \
        "{{ '<start_of_turn>user\\n' + message['value'] }}" \
    "{% elif message['from'] == 'human' %}" \
        "{% if loop.index0 == 1 %}" \
            "{{ '\\nUser Question:\\n' }}" \
        "{% else %}" \
            "{{ '<start_of_turn>user\\n' }}" \
        "{% endif %}" \
        "{{ message['value'] + '<end_of_turn>' }}" \
    "{% elif message['from'] == 'gpt' %}" \
        "{{ '<start_of_turn>model\\n'  + message['value'] + ' ' + '<end_of_turn>' }}" \
    "{% elif message['from'] == 'function_response' %}" \
        "{{ '<start_of_turn>user\\n'  + message['value'] + ' ' + '<end_of_turn>' }}" \
    "{% endif %}" \
    "{% if not loop.last %}" \
        "{{ '\\n' }}" \
    "{% endif %}" \
"{% endfor %}" \
"{% if not add_generation_prompt is defined %}" \
    "{% set add_generation_prompt = false %}" \
"{% endif %}" \
"{% if add_generation_prompt %}" \
    "{{ '\\n<start_of_turn>model\\n' }}" \
"{% endif %}"

tokenizer.chat_template = chat_template

dataset_train = load_dataset("D:\\Code\\models\\function-calling-sharegpt", split="train")
dataset_train = dataset_train.map(
    lambda x: {
        "formatted_chat": tokenizer.apply_chat_template(x["conversation"], tokenize=False)
    }
)

model = AutoModelForCausalLM("Baichuan2-7B-Chat", attn_implementation="flash_attention_2")

model.config.use_cache = False
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["W_pack"],
    inference_mode=False,
    r=1,
    lora_alpha=32,
    lora_dropout=0.1
)

model.enable_input_require_grad()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    fp16=False,
    bf16=True,
    save_strategy="epoch",
    logging_steps=1,
    learning_rate=2e-4,
    group_by_length=True,
    lr_scheduler_type="constant",
    optim="adamw_torch",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset_train, tokenizer=tokenizer
)

trainer.train()
trainer.save_state()
trainer.save_model("baichuan2-7b-chat-function-calling")

