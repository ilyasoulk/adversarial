import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from tqdm import tqdm

data_path = "dataset/exercises.json"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# We setup the LoRA configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We setup the LoRA configuration

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# We import model and tokenizer

checkpoint = "microsoft/phi-2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    checkpoint, device_map={"": device}, peft_config=peft_config
)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, padding_side="left", pad_token='<pad>'
)

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

print_trainable_parameters(model)

ppo_config = PPOConfig(
    model_name="microsoft/phi-2",
    learning_rate=1e-5,
    remove_unused_columns=False,
    batch_size=8,
    mini_batch_size=2
    # More hyperparameters can be added here
)

def preprocess_function(examples):
    input_ids = []
    attention_masks = []
    unit_tests = []
    prefix = '''Instruct: For the following function please only generate the solution and only the solution.
    After you finish generating the function asked you should stop the generation. The python code starts with a ```python
 and once you finish generating the function you should close the code with a ```.\nOutput:\n```python\n
    '''

    for docstring, unit_test in zip(examples['docstring'], examples['unit_tests']):
        prompt = prefix + '\n' + docstring
        tokenized = tokenizer(prompt, truncation=True)
        input_ids.append(torch.tensor(tokenized['input_ids']))
        unit_tests.append(unit_test)

    return {
        "input_ids": input_ids,
        "unit_tests": unit_tests
    }

def data_collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

dataset = load_dataset('json', data_files=data_path, split='train')

ds = dataset.map(
    preprocess_function,
    batched=True
)

ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

ds.set_format(type="torch")

trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
    dataset=ds,
    data_collator=data_collator
)

generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
    "max_new_tokens": 200,
    "early_stopping": True,
}

def extract_code(text, occurrence=1):
    # Using regular expression to find all occurrences of text enclosed in triple backticks
    codes = re.findall(r'```python(.*?)```', text, re.DOTALL)
    if len(codes) > occurrence:
        return codes[occurrence].strip()
    else:
        print('We could not parse the response')
        return ''



epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(trainer.dataloader, "batch: "):
        query_tensors = batch['input_ids']

        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["response"] = [extract_code(r) for r in batch["response"]]

        texts = [r for r in batch['response']]
        unit_tests = [tests for tests in batch['unit_tests']]
        rewards = [torch.tensor(reward_model(code, tests)) for code, tests in zip(texts, unit_tests)]
        print(rewards)

        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards)