from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

# Set your own token or use another method like login()
token = 'your_token_here';

# checkpoint = "bigcode/starcoder"
checkpoint = "bigcode/starcoderbase-3b"
# checkpoint = "codellama/CodeLlama-7b-hf"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model_star = AutoModelForCausalLM.from_pretrained(checkpoint, token=token)
# model_llama = AutoModelForCausalLM.from_pretrained(llama, token=token)



model_star = get_peft_model(model_star, peft_config)
# model_llama = get_peft_model(model_llama, peft_config)


print('StarCoder model')
model_star.print_trainable_parameters()
# print('Llama model')
# model_llama.print_trainable_parameters()