from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

token = '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';

checkpoint = "bigcode/starcoder"

# llama = "codellama/CodeLlama-7b-hf"
# device = torch.device("cpu")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model_star_3b = AutoModelForCausalLM.from_pretrained(checkpoint, token=token)
# model_llama = AutoModelForCausalLM.from_pretrained(llama, token=token)



model_star_3b = get_peft_model(model_star_3b, peft_config)
# model_llama = get_peft_model(model_llama, peft_config)


print('Star 3b model')
model_star_3b.print_trainable_parameters()
# print('Llama model')
# model_llama.print_trainable_parameters()