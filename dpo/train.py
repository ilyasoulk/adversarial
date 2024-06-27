import torch
import numpy as np
from datasets import load_dataset
import argparse
from generate_dataset import (
    create_dataset,
    create_pipeline,
    load_json,
)
from transformers import AutoTokenizer
from peft import PeftModel
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from trl import DPOTrainer
from peft import LoraConfig
from transformers import TrainingArguments
import gc
from parse_args import parse_args


def get_train_dataset(dataset_path: str, tokenizer):
    ds = load_dataset("json", data_files=dataset_path, split="train")

    def transform_to_conversation(prompt, response):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    def process(row):
        chosen_conversation = transform_to_conversation(row["prompt"], row["chosen"])
        row["chosen"] = tokenizer.apply_chat_template(
            chosen_conversation, tokenize=False
        )
        rejected_conversation = transform_to_conversation(
            row["prompt"], row["rejected"]
        )
        row["rejected"] = tokenizer.apply_chat_template(
            rejected_conversation, tokenize=False
        )
        return row

    train_dataset = ds.map(
        process,
        load_from_cache_file=False,
    )
    return train_dataset


def get_hard_exercises(dpo_trainer, n_samples):
    data = dpo_trainer.get_train_dataloader()

    exercises = []
    scores = []
    print(len(data))

    for batch in data:
        recap = dpo_trainer.get_batch_loss_metrics(dpo_trainer.model, batch)
        reward = recap[1]["rewards/margins"].item()
        for exercise in batch["prompt"]:
            exercises.append(exercise)
            scores.append(reward)
        del recap
        gc.collect()

    # Extract the second element from each tuple for softmax calculation
    scores = np.array(scores)
    exp_scores = np.exp(-scores)  # apply negative exponent to invert the scores
    probabilities = exp_scores / np.sum(
        exp_scores
    )  # normalize to create a probability distribution

    # Sample n_samples from the distribution
    hard_exercises = np.random.choice(
        exercises, size=n_samples, p=probabilities, replace=False
    )
    return hard_exercises


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parse_args(parser)

    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_cuda_devices}")

    print("Creating pipelines...")
    if num_cuda_devices <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oracle = create_pipeline(args.oracle_path, device)
        if args.student_path != args.oracle_path:
            student = create_pipeline(args.student_path, device)
        else:
            student = oracle
    else:
        print("Oracle and student models are on different devices.")
        oracle = create_pipeline(
            args.oracle_path, device="cuda:0", do_quantization=args.do_quantization
        )

        print(f"Oracle model is on device {oracle.device}")
        student = create_pipeline(
            args.student_path, device="cuda:1", do_quantization=args.do_quantization
        )
        print(f"Student model is on device {student.device}")

    professions = load_json("tree/professions.json")
    topics_list = load_json("tree/topics.json")

    print("Creating dataset...")
    if args.use_existing_dataset:
        dataset_path = args.dataset_path
    else:
        dataset_path = create_dataset(
            oracle,
            student,
            topics_list,
            professions,
            args.num_subtopic_per_topic,
            args.num_exercise_per_subtopic,
            args.dataset_path,
            args.oracle_max_length,
            args.oracle_temperature,
            args.student_max_length,
            args.student_temperature,
        )
    del student
    gc.collect()
    # If the student pipeline is on a different device we could maybe call torch.cuda.empty_cache() here

    student_path = args.student_path

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(student_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Truncate from left to ensure we don't lose labels in final turn
    # tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # specify how to quantize the model
    if args.do_quantization:
        print("Setting up quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    device_map = (
        {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_map = {f"cuda:{i}": i for i in range(torch.cuda.device_count())}
    else:
        device_map = None

    # Step 1: load the base model (Mistral-7B in our case) in 4-bit
    model_kwargs = dict(
        # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False,  # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config if args.do_quantization else None,
    )

    training_args = TrainingArguments(
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        max_steps=args.num_steps,
        save_strategy="no",
        logging_steps=1,
        output_dir=args.output_dir,
        optim="paged_adamw_32bit",
        warmup_steps=100,
        report_to="tensorboard",
        seed=42,
    )

    # based on the recipe: https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_qlora.yaml
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    for i in range(args.repetitions):
        print(f"Starting iteration {i}")
        # If it's not the first iteration we don't need to generate the dataset since we did it at the end of the previous iteration
        dataset = get_train_dataset(dataset_path=dataset_path, tokenizer=tokenizer)
        print(f"Dataset of size {len(dataset)} generated")

        print("Loading the model...")
        model = AutoModelForCausalLM.from_pretrained(student_path, **model_kwargs)

        trainer = DPOTrainer(
            model,
            None,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            beta=args.beta,
            max_prompt_length=1024,
            max_length=1536,
        )

        print("Training the model...")
        train_result = trainer.train()

        # Save the adapters
        print("Saving the model...")
        trainer.model.save_pretrained(f"checkpoint_v{i}")
        trainer.tokenizer.save_pretrained(f"checkpoint_v{i}")
        reference_exercises = get_hard_exercises(trainer)

        # Flush memory
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        # Load the base model
        print("Loading the base model for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            student_path, return_dict=True, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(student_path)

        # Merge the adapters into the base model
        print("Merging the adapters...")
        model = PeftModel.from_pretrained(base_model, f"checkpoint_v{i}")
        model = model.merge_and_unload()

        # Save the model
        student_path = f"model_v{i}"
        model.save_pretrained(student_path)
        tokenizer.save_pretrained(student_path)
        # If this is not the last iteration, generate a new dataset
        if i != args.repetitions - 1:
            print("Generating adversarial dataset...")
            student = create_pipeline(student_path, device)
            dataset_path = create_dataset(
                oracle,
                student,
                topics_list,
                professions,
                args.num_subtopic_per_topic,
                args.num_exercise_per_subtopic,
                args.dataset_path,
                args.oracle_max_length,
                args.oracle_temperature,
                args.student_max_length,
                args.student_temperature,
                reference_exercises,
            )

            del student
            gc.collect()
