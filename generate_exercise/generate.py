import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os

device = torch.device("cpu")
token = os.environ.get("TOKEN")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5",
                                             torch_dtype=torch.float32,
                                             trust_remote_code=True, token=token).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5",
                                          trust_remote_code=True, token=token)

current_working_directory = os.getcwd()
output_directory_name = "generated_exercises"
output_dir = os.path.join(current_working_directory, output_directory_name)
os.makedirs(output_dir, exist_ok=True)

test_2 = '''Generate a list of unique and challenging coding exercises focused on enhancing algorithmic thinking and problem-solving skills in Python. 
Each exercise should test different core programming concepts, such as recursion, sorting algorithms, data structures (lists, dictionaries, trees), string manipulation, and search algorithms.
Ensure the exercises are clear and have a defined objective, similar to the example provided:

1. Write a function that finds the maximum depth of list nesting in a given list.

Please include a brief description of the problem and specify the input and output formats. 
Aim for a variety of difficulties, from beginner to advanced levels, to cater to a wide range of programming expertise.'''

prompt = tokenizer(test_2, return_tensors="pt", truncation=True, max_length=1024)
outputs = model.generate(**prompt, temperature=0.9, do_sample=True,
                         max_length=500, eos_token_id=tokenizer.convert_tokens_to_ids(['<|endoftext|>']), top_p=0.95)
text = tokenizer.batch_decode(outputs)[0]

print(text)

pattern = r"(Question\s+\d+:\s+|\d+\.\s+)([^\n]+)"
# pattern = r"\bWrite\b[^.]*\."
matches = re.findall(pattern, text, re.DOTALL)
print("Matches: ", matches)
k = 1
print("\n\n")
matches.pop(0)  # Remove the example from the prompt
for match in matches:
    print("Match: ", match)
    question = match[1]
    with open(os.path.join(output_dir, f"question_{k}.txt"), "w") as file:
        file.write(question + "\n")
    k += 1


for question in os.listdir(output_dir):
    with open(os.path.join(output_dir, question), "a+") as file:
        question = file.read()
        test_3 = f'''Please generate a series of Python assert statements to serve as unit tests for the following coding exercise.
Each assert statement should test a specific aspect of the function's behavior, such as handling normal cases, edge cases, and any special conditions outlined in the exercise description. Provide a brief explanation for each assert statement to clarify what it tests and why it's important. Here's the coding exercise:

Coding Exercise Description:
{question}

Requirements:

The series of assert statements should comprehensively test the function, covering various input scenarios including normal cases, edge cases, and error handling (if applicable).
Include comments with each assert statement to explain the intent behind the test and what it verifies.
Ensure the tests are straightforward to understand and execute, avoiding complex testing frameworks or external dependencies.
Please format the assert statements so they can be executed directly within a Python script or interactive session, ensuring a simple and effective testing approach.'''

        prompt_2 = tokenizer(test_3, return_tensors="pt", truncation=True, max_length=1024)
        outputs_2 = model.generate(**prompt_2, temperature=0.9, do_sample=True,
                                 max_length=700, eos_token_id=tokenizer.convert_tokens_to_ids(['<|endoftext|>']),
                                   top_p=0.95)

        text_2 = tokenizer.batch_decode(outputs_2)[0]

        print(text_2)
        pattern = r"^\s*assert\s+.+"
        matches = re.findall(pattern, text_2, re.MULTILINE)
        print("Matches: ", matches)
        for match in matches:
            print("Match: ", match)
            file.write(match + "\n")