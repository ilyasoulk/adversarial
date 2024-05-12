import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from reward_model import reward_model
from trl import PPOConfig, PPOTrainer

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
    sample['input_ids'] = tokenizer(sample['docstring'], return_tensors="pt", truncation=True, max_length=512)['input_ids']
    return sample

# TODO : Import the dataset, use it for the PPO config. Implement the training loop