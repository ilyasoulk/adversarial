import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

dataset = load_dataset("openai_humaneval")
test_dataset = dataset["test"]

device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5",
                                             torch_dtype=torch.float32,
                                             trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5",
                                          trust_remote_code=True)
number_of_tests = 1
successes = 0
pass_k = 3

for i in range(number_of_tests):
    prompt = test_dataset[i]["prompt"]
    test = test_dataset[i]["test"]
    entry_point = test_dataset[i]["entry_point"]
    task_id = test_dataset[i]["task_id"]
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512)
    eos_token_id = tokenizer.eos_token_id  # Get the end-of-sequence token ID
    outputs = model.generate(**inputs, max_length=250,
                             early_stopping=True, num_beams=8,
                             eos_token_id=tokenizer.convert_tokens_to_ids(['<|endoftext|>']), num_return_sequences=pass_k)
    generated_solutions = tokenizer.batch_decode(outputs)
    for k in range(generated_solutions):
        lines = generated_solutions[k].split('\n')
        lines = lines[:-1]
        modified_solution = '\n'.join(lines)
        script = modified_solution + '\n' + test + f'check({entry_point})'
        print("Script : \n" + script)
        try:
            exec(script)
            print(GREEN + f'{task_id} : OK' + RESET)
            print(f'Passed at k = {k}')
            successes += 1
        except SyntaxError as e:
            print(RED + f'{task_id} : FAIL : {e}' + RESET)
        except AssertionError:
            print(RED + f'{task_id} : FAIL' + RESET)


success_rate = (successes / number_of_tests) * 100
color = GREEN if success_rate >= 50 else RED
print(f'{color}Success rate : {success_rate}%{RESET}')
