import torch
from datasets import load_dataset
import argparse
from generate_dataset import (
    create_dataset,
    create_pipeline,
    load_json,
)
from transformers import AutoTokenizer
from peft import PeftConfig
import re
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from trl import DPOTrainer
from peft import LoraConfig
from transformers import TrainingArguments
import heapq
from datasets import DatasetDict
import gc
from multiprocessing import cpu_count


def apply_chat_template(example, tokenizer, assistant_prefix="<|assistant|>\n"):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("chosen", "rejected")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [
            [msg for msg in example["chosen"] if msg["role"] == "user"][0]
        ]
        # Insert system message
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["chosen"][0])
        # TODO: handle case where chosen/rejected also have system messages
        chosen_messages = example["chosen"][1:]
        rejected_messages = example["rejected"][1:]
        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False
        )
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False
        )
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        example["text_rejected"] = _strip_prefix(
            example["text_rejected"], assistant_prefix
        )
    else:
        raise ValueError(
            f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )

    return example


def get_hard_exercises(dpo_trainer):
    data = dpo_trainer.get_train_dataloader()

    exercises_rankings = []

    for batch in data:
        recap = dpo_trainer.get_batch_loss_metrics(dpo_trainer.model, batch)
        exercises_rankings.append((batch["prompt"], recap[1]["rewards/chosen"].item()))
        del recap
        gc.collect()

    hardest_exercises = heapq.nsmallest(
        4, exercises_rankings, key=lambda x: x[1]
    )  # If the batch size is 4, we should be able to retrieve 16 exercises
    return [exercise[0] for exercise in hardest_exercises]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle_path",
        type=str,
        required=True,
        help="Path to the oracle model",
    )
    parser.add_argument(
        "--student_path",
        type=str,
        required=True,
        help="Path to the student model",
    )
    parser.add_argument(
        "--num_subtopic_per_topic",
        type=int,
        default=3,
        help="Number of subtopics per topic",
    )
    parser.add_argument(
        "--num_exercise_per_subtopic",
        type=int,
        default=3,
        help="Number of exercises per subtopic",
    )
    parser.add_argument(
        "--oracle_temperature",
        type=float,
        default=1.0,
        help="Temperature for oracle text generation",
    )
    parser.add_argument(
        "--oracle_max_length",
        type=int,
        default=2500,
        help="Maximum length of oracle generation",
    )
    parser.add_argument(
        "--student_temperature",
        type=float,
        default=0.6,
        help="Temperature for student text generation",
    )
    parser.add_argument(
        "--student_max_length",
        type=int,
        default=512,
        help="Maximum length of student generation",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Output path for the generated dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5.0e-6,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Beta for DPO training",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = create_pipeline(args.oracle_path, device)
    student = create_pipeline(args.student_path, device)
    professions = load_json("tree/professions.json")
    topics_list = load_json("tree/topics.json")
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

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    student_path = args.student_path

    for i in range(args.repetitions):
        # If it's not the first iteration we don't need to generate the dataset since we did it at the end of the previous iteration
        size = dataset.num_rows
        # Split the dataset into train and test
        train_set = dataset.select(range(int(size * 0.8)))
        test_set = dataset.select(range(int(size * 0.8), size))
        dataset_dict = {"train": train_set, "test": test_set}
        dataset = DatasetDict(dataset_dict)
        tokenizer = AutoTokenizer.from_pretrained(student_path)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Truncate from left to ensure we don't lose labels in final turn
        tokenizer.truncation_side = "left"

        # Set reasonable default for models without max length
        if tokenizer.model_max_length > 100_000:
            tokenizer.model_max_length = 2048

        DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        column_names = list(dataset["train"].features)

        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=cpu_count(),
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )

        # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
        for split in ["train", "test"]:
            dataset[split] = dataset[split].rename_columns(
                {
                    "text_prompt": "prompt",
                    "text_chosen": "chosen",
                    "text_rejected": "rejected",
                }
            )

        peft_config = PeftConfig.from_pretrained(student_path)

        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        device_map = (
            {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        )

        # Step 1: load the base model (Mistral-7B in our case) in 4-bit
        model_kwargs = dict(
            # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
            torch_dtype="auto",
            use_cache=False,  # set to False as we're going to use gradient checkpointing
            device_map=device_map,
            quantization_config=quantization_config,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, **model_kwargs
        )
        # Step 2: load base model + SFT adapter weights
        # notice that only the adapter weights are trainable!
        model = PeftModel.from_pretrained(base_model, student_path)

        # based on config
        training_args = TrainingArguments(
            bf16=True,
            beta=0.01,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=100,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            hub_model_id="zephyr-7b-dpo-qlora",
            learning_rate=5.0e-6,
            log_level="info",
            logging_steps=10,
            lr_scheduler_type="cosine",
            max_length=1024,
            max_prompt_length=512,
            num_train_epochs=1,
            optim="paged_adamw_32bit",
            output_dir=args.output_dir,  # It is handy to append `hub_model_revision` to keep track of your local experiments
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            # push_to_hub=True,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            seed=42,
            warmup_ratio=0.1,
        )

        # based on the recipe: https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_qlora.yaml
        peft_config = LoraConfig(
            r=128,
            lora_alpha=128,
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

        trainer = DPOTrainer(
            model,
            ref_model=None,
            model_init_kwargs=None,
            ref_model_init_kwargs=None,
            args=training_args,
            beta=training_args.beta,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            max_prompt_length=training_args.max_prompt_length,
            peft_config=peft_config,
            loss_type=training_args.loss_type,
        )

        train_result = trainer.train()

        metrics = train_result.metrics
        max_train_samples = (
            training_args.max_train_samples
            if training_args.max_train_samples is not None
            else len(dataset["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(dataset["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # Flush memory

        if i != args.repetitions - 1:
            reference_exercises = get_hard_exercises(trainer)
            del trainer
            gc.collect()
            student_path = args.output_dir

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

            dataset = load_dataset("json", data_files=dataset_path, split="train")
            del student
            gc.collect()
