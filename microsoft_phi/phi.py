import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import json

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


current_working_directory = os.getcwd()
output_directory_name = "generated_outputs"
output_dir = os.path.join(current_working_directory, output_directory_name)
os.makedirs(output_dir, exist_ok=True)

number_of_tests = 1
successes = 0
pass_k = 1

def generate_outputs(test_dataset, model, tokenizer, number_of_tests , pass_k, output_dir):
    print("Starting to generate outputs")
    for i in range(number_of_tests):
        prompt = test_dataset[i]["prompt"]
        evaluation = test_dataset[i]["test"]
        entry_point = test_dataset[i]["entry_point"]
        task_id = test_dataset[i]["task_id"]
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=250,
                             early_stopping=True, num_beams=8,
                             eos_token_id=tokenizer.convert_tokens_to_ids(['<|endoftext|>']), num_return_sequences=pass_k)
        generated_solutions = tokenizer.batch_decode(outputs)

        print(f"Task ID: {task_id}")
        dirname = f"HumanEval_{i}"
        input_dir = os.path.join(output_dir, dirname)
        os.makedirs(input_dir, exist_ok=True)

        for k, solution in enumerate(generated_solutions):
            lines = solution.split('\n')
            lines = lines[:-1]
            modified_solution = '\n'.join(lines)
            with open(os.path.join(input_dir, f"{dirname}_{k}"), "w") as file:
                file.write(modified_solution + f'\n{evaluation}\ncheck({entry_point})')


def run_tests(output_dir):
    results = []
    for task_id in os.listdir(output_dir):
        task_dir = os.path.join(output_dir, task_id)
        files = os.listdir(task_dir)
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1]))
        k = 0
        test_passed = False
        for file in sorted_files:
            file_path = os.path.join(task_dir, file)
            with open(file_path, "r") as file:
                solution = file.read()
                try:
                    exec(solution)
                    print(f"{GREEN}Test passed at k = {k}{RESET}")
                    test_passed = True
                    break
                except Exception as e:
                    print(RED + f'Unexpected error: {e}' + RESET)
                finally:
                    k += 1

        results.append({"task_id": task_id, "passed": test_passed, "attempts": k})
    results_file_path = os.path.join(output_dir, "test_results.json")
    with open(results_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


generate_outputs(test_dataset, model, tokenizer, number_of_tests, pass_k, output_dir)
run_tests(output_dir)

