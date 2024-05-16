import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from reward_model import reward_model
from trl import PPOConfig, PPOTrainer
from datasets import load_dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "dataset/exercises.json"
dataset = torch.load(data_path) # A list of dictionaries containing the docstring and unit tests
exercises = [tokenize(sample['docstring']) for sample in dataset]

# We setup the LoRA configuration

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# We import model and tokenizer

checkpoint = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# We setup the PPO configuration

ppo_config = PPOConfig(
    model_name="microsoft/phi-1_5",
    learning_rate=1e-5,
    # More hyperparameters can be added here
)

def tokenize(sample):
    sample['input_ids'] = tokenizer.encode(sample['docstring'], return_tensors='pt')
    return sample
# TODO : Import the dataset, use it for the PPO config. Implement the training loop

dataset = load_dataset('json', 'dataset/exercises.json')
dataset = dataset.map(tokenize, batched=False)

trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
    device=device,
    dataset=dataset
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(trainer.train_dataloader, "batch: "):
        query_tensors = batch['input_ids'].to(device)

        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [r for r in batch['response']]
        unit_tests = [tests for tests in batch['unit_tests']]
        rewards = [reward_model(code, tests) for code, tests in zip(texts, unit_tests)]

        stats = trainer.train_step(batch, rewards)

trainer.save_pretrained("ppo_trained_model")