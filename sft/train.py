import os
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import re
import random
from multiprocessing import cpu_count

def concatenate_json_lists(directory):
    # Initialize an empty list to store all JSON data
    all_data = []

    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Read and parse each JSON file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    
                    # Ensure the loaded data is a list
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        print(f"Warning: {filename} does not contain a JSON list. Skipping.")
                except json.JSONDecodeError:
                    print(f"Error: {filename} is not a valid JSON file. Skipping.")

    return all_data


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


if __name__ == "__main__":
    dataset_path = "datasets/dataset_mistral7b_phi_sft.json"
    dataset = load_dataset("json", data_files=dataset_path, split='train')
    print(dataset)

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    # Set chat template
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    column_names = list(dataset.features)
    dataset = dataset.map(apply_chat_template,
                                    num_proc=cpu_count(),
                                    fn_kwargs={"tokenizer": tokenizer},
                                    remove_columns=column_names,
                                    desc="Applying chat template",)


    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed training set:\n\n{dataset[index]['text']}")

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    model_kwargs = dict(
        # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
    )

        # path where the Trainer will save its checkpoints and logs
    output_dir = 'models/qwen2.5-3b-instruct-sft-mistral-7b'

    # based on config
    training_args = TrainingArguments(
        bf16=True, # specify bf16=True instead when training on GPUs that support bf16
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2.0e-05,
        log_level="info",
        logging_steps=5,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        max_steps=-1,
        num_train_epochs=5,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=1, # originally set to 8
        per_device_train_batch_size=1, # originally set to 8
        save_strategy="epoch",
        save_total_limit=None,
        report_to="wandb",
        seed=42,
    )

    # based on config
    peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = SFTTrainer(
            model=model_id,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
            packing=True,
            peft_config=peft_config,
            max_seq_length=tokenizer.model_max_length,
        )

    train_result = trainer.train()

    trainer.save_model(output_dir)
