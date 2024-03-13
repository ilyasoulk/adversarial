from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
import os

# Set your own token or use another method like login()
token = os.environ.get("TOKEN")
# checkpoint = "bigcode/starcoderbase-1b"
checkpoint = "microsoft/phi-2"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


model = AutoModelForCausalLM.from_pretrained(checkpoint, token=token)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
