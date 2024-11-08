import argparse
import os
import time
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import json
from itertools import islice
import time
import json
import argparse
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List
from pydantic import BaseModel
from datasets import load_dataset


class Topic(BaseModel):
    topic: str


class Exercise(BaseModel):
    exercise: str
    topic: Topic


class Query(BaseModel):
    query: str
    topic: Topic


def extract_list_from_string(text):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if len(matches) > 0:  # Check if there's at least 2 matches
        list_content = matches[0]  # Get the second match
        list_items = [
            item.strip().strip('"').strip("'") for item in list_content.split(",")
        ]
        
        return list_items
    return []


def create_prompt_query(topic: Topic, profession: str, n: int) -> str:
    query = f'''
            Create {n} unique and challenging Python code completion exercises on the topic of “{topic}”.
            Each exercise should target the skill level of a {profession} and provide a complete solution.

            Structure each exercise as follows:
            
            ```
            def function_name(parameters):
                """
                Description of the task or problem to solve, providing all necessary context.
                """
                # Solution code starts here (in Python)
            ```
            
            Guidelines:
            - Avoid using classes.
            - Make each exercise very difficult to challenge the user.
            - Ensure a complete solution is provided immediately following the docstring.
            '''
    query = "\n".join([line.strip() for line in query.strip().split("\n")])
    return query


def create_prompt(topic: Topic, professions: List[str], n: int) -> Query:

    profession = professions[np.random.randint(0, len(professions))]
    query = create_prompt_query(topic, profession, n)

    return query


def create_topic_prompts(topics, n, reference_exercises=[]):
    res = []
    if len(reference_exercises) > 0: # Adversarial episode
        for reference in reference_exercises:
            # Incremental Complexity Prompt
            incremental_complexity_prompt = f"""
            Based on the reference exercise '{reference}', generate {n} new coding exercises that introduce additional edge cases and increased complexity.
            
            Requirements:
            - Build upon the original task, adding nuanced complexity.
            - Avoid using classes, but require complex logic and multiple edge cases.
            - Provide a solution following the exercise.

            Format:
            ```
            def function_name(parameters):
                \"\"\"Exercise description with additional complexity.\"\"\"
                # Solution code here
            ```
            """

            # Opposite Approach Prompt
            opposite_approach_prompt = f"""
            Create {n} adversarial coding exercises for the topic related to '{reference}' that challenge conventional assumptions made in the original exercise.
            
            Requirements:
            - Shift expected assumptions, requiring the model to adapt to unexpected scenarios.
            - Keep exercises challenging and avoid using classes.
            - Provide solutions immediately after each exercise.

            Format:
            ```
            def function_name(parameters):
                \"\"\"Exercise with altered assumptions.\"\"\"
                # Solution code here
            ```
            """

            # Deceptive Complexity Prompt
            deceptive_complexity_prompt = f"""
            Generate {n} coding exercises inspired by '{reference}' that appear simple but involve hidden complexities.
            
            Requirements:
            - Exercises should look straightforward but require complex solutions.
            - Avoid classes and focus on deceptive problem setups.
            - Solutions should immediately follow the exercise.

            Format:
            ```
            def function_name(parameters):
                \"\"\"Exercise with hidden complexities.\"\"\"
                # Solution code here
            ```
            """

            # Conceptual Shift Prompt
            conceptual_shift_prompt = f"""
            Create {n} exercises on a topic related to '{reference}' that require alternative reasoning paths.
            
            Requirements:
            - Exercises should challenge conventional solutions, pushing the model to apply different reasoning approaches.
            - Keep exercises difficult and avoid classes.
            - Provide solutions directly after each exercise.

            Format:
            ```
            def function_name(parameters):
                \"\"\"Exercise requiring alternative reasoning.\"\"\"
                # Solution code here
            ```
            """

            # Error-Based Prompt
            error_based_prompt = f"""
            Based on common errors found in solving the exercise '{reference}', generate {n} new exercises that incorporate these common pitfalls.
            
            Requirements:
            - Design exercises to test the model’s ability to avoid or recognize typical mistakes.
            - Keep exercises challenging and avoid using classes.
            - Solutions should follow each exercise.

            Format:
            ```
            def function_name(parameters):
                \"\"\"Exercise addressing common pitfalls.\"\"\"
                # Solution code here
            ```
            """
            
            # Append each adversarial prompt for the given reference exercise
            res.extend([
                incremental_complexity_prompt,
                opposite_approach_prompt,
                deceptive_complexity_prompt,
                conceptual_shift_prompt,
                error_based_prompt
            ])

    else:
        for topic in topics:
            prompt = f"""
            As a Python textbook author, create {n} distinct subtopics for the main topic '{topic}'. 

            Requirements:
            - Each subtopic should be broad enough to generate multiple exercise types (theory, practice, debugging).
            - Include both fundamental and advanced concepts.
            - Ensure topics build on each other in a logical learning progression.
            - Mix conceptual and practical application areas.
            - Avoid overlapping subtopics.

            Format your response as a Python list of strings, like this:
            ['Basic String Operations', 'String Formatting Methods', 'Regular Expressions']

            Return only the Python list with no additional text or explanation.
            """
            res.append(prompt)

    return res


def generate(model, tokenizer, batch_size, prompts, generation_config, desc="Generating...", instruct=False):
    responses = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i:i + batch_size]
        
        if instruct:
            inputs = tokenizer.apply_chat_template(batch, return_tensors="pt", padding=True, truncation=True, return_dict=True)
        else:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        inputs = inputs.to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config,
                num_return_sequences=1,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
        # Decode
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)
        
        print(f"Processed {i + len(batch)}/{len(prompts)} prompts")

    return responses


def extract_functions(text):
    """
    Extracts both function definitions (with docstrings) and their implementations from Python code,
    excluding the first function occurrence.
    
    Args:
        text (str): Input text containing Python function definitions
        
    Returns:
        list: List of dictionaries containing function definitions and implementations
    """
    # Pattern to match complete functions including:
    # 1. Function definition (def name(args):)
    # 2. Docstring
    # 3. Implementation code
    pattern = r'(def\s+(\w+)\([^)]*\):)\s*("""(?:(?!""")[\s\S])*?""")\s*((?:(?!def\s+\w+\([^)]*\):)[\s\S])*?)(?=\s*def\s+\w+\([^)]*\):|\s*$)'
    
    # Find all matches
    matches = list(re.finditer(pattern, text))
    
    # Skip the first match if it exists
    
    # Store functions
    functions = []
    
    for match in matches:
        func_def = match.group(1)
        func_name = match.group(2)
        docstring = match.group(3)
        implementation = match.group(4).strip()
        
        # Clean up implementation
        implementation = re.sub(r'^\s*\n', '', implementation)
        implementation = re.sub(r'\n\s*$', '', implementation)
        
        # Remove common indentation from implementation
        lines = implementation.split('\n')
        if lines:
            # Find minimum indentation (excluding empty lines)
            indentations = [len(line) - len(line.lstrip()) 
                          for line in lines if line.strip()]
            if indentations:
                min_indent = min(indentations)
                # Remove the common indentation from each line
                implementation = '\n'.join(
                    line[min_indent:] if line.strip() else ''
                    for line in lines
                )
        
        functions.append({
            'func_def': f"{func_def}\n    {docstring}",
            'code': implementation,
        })


    print(f"Extracted {len(functions)} functions")
    
    return functions


def create_prompts_dataset(oracle, or_tokenizer, topics, num_subtopics, reference_exercises, professions, num_exercises, batch_size, generation_config):
    subtopics_list = []
    prompts = create_topic_prompts(topics, num_subtopics, reference_exercises=reference_exercises)
    if len(reference_exercises) == 0:
        topics_raw = generate(oracle, or_tokenizer, batch_size=batch_size, prompts=prompts, generation_config=generation_config, desc="Generating Topics...")
        for topic_raw in topics_raw:
            subtopics_list.extend(
                extract_list_from_string(topic_raw)
            )

        with open("tree/subtopics.json", "w") as f:
            json.dump(subtopics_list, f, indent=4)



        # Create prompts
        queries = [
            create_prompt(subtopic, professions, num_exercises)
            for subtopic in subtopics_list
        ]

    else:
        queries = prompts

    with open("tree/prompts.json", "w") as f:
        json.dump(queries, f, indent=4)

    return queries


def create_dataset(
    oracle,
    or_tokenizer,
    student,
    st_tokenizer,
    topics,
    professions,
    num_subtopics,
    num_exercises,
    dataset_path,
    generation_config_oracle,
    generation_config_student,
    reference_exercises=[],
    batch_size=16,
):
    # Generate subtopics (same as before)


    queries = create_prompts_dataset(oracle, or_tokenizer, topics, num_subtopics, reference_exercises, professions, num_exercises, batch_size, generation_config_oracle)

    # Convert queries to proper chat format
    messages_dataset = [
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": query}
        ] for query in queries
    ]

    # Generate both prompts and chosen
    responses = generate(oracle, or_tokenizer, batch_size, messages_dataset, generation_config=generation_config_oracle, desc="Generating DPO Prompts", instruct=True)
    dataset = []
    for response in responses:
        dataset.extend(extract_functions(response))

    # Generate rejecteds
    dpo_prompts = [f"Complete the following python code :\n{data["func_def"]}" for data in dataset]
    winning = generate(oracle, or_tokenizer, batch_size, dpo_prompts, generation_config=generation_config_oracle, desc="Generating Winning Solutions")
    rejected = generate(student, st_tokenizer, batch_size, dpo_prompts, generation_config=generation_config_student, desc="Generating Rejected Solutions")

    clean_dataset = []
    # Clean up and compose dataset
    for prompt, win, reject in zip(dpo_prompts, winning, rejected):
        # If empty we ignore
        if prompt == "" or win == "" or reject == "":
            continue

        clean_dataset.append({"prompt" : prompt, "chosen": win, "rejected": reject})


    with open(dataset_path, "w") as f:
        json.dump(clean_dataset, f, indent=4)

    return dataset


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return None


def load_model(model_name, do_quantization=False):
    if do_quantization:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quantization_config if do_quantization else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model.generation_config.cache_implementation = "static"
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    return model, tokenizer


def dataset_generation(oracle_path, student_path, args):

    oracle, or_tokenizer = load_model(oracle_path)
    student, st_tokenizer = load_model(student_path)

    professions = load_json("tree/professions.json")
    topics_list = load_json("tree/topics.json")

    print("Creating dataset...")
    start_time = time.time()
    generation_config_oracle = {"temperature": args.oracle_temperature, "max_length": args.oracle_max_length}
    generation_config_student = {"temperature": args.student_temperature, "max_length": args.student_max_length}
    
    create_dataset(
        oracle,
        or_tokenizer,
        student,
        st_tokenizer,
        topics_list,
        professions,
        args.num_subtopic_per_topic,
        args.num_exercise_per_subtopic,
        args.dataset_path,
        generation_config_oracle,
        generation_config_student,
    )
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset using text generation models."
    )
    parser.add_argument(
        "--oracle_path",
        type=str,
        required=True,
        help="File path or HF path for the oracle model",
    )
    parser.add_argument(
        "--student_path",
        type=str,
        required=True,
        help="File path or HF path for the student model",
    )
    parser.add_argument(
        "--num_subtopic_per_topic",
        type=int,
        default=10,
        help="Number of subtopics per topic",
    )
    parser.add_argument(
        "--num_exercise_per_subtopic",
        type=int,
        default=5,
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

    args = parser.parse_args()

    dataset_generation(args.oracle_path, args.student_path, args)

    # num_cuda_devices = torch.cuda.device_count()
    # print(f"Number of CUDA devices: {num_cuda_devices}")

    # print("Creating pipelines...")

    # oracle, or_tokenizer = load_model(args.oracle_path)
    # student, st_tokenizer = load_model(args.student_path)

    # professions = load_json("tree/professions.json")
    # topics_list = load_json("tree/topics.json")

    # print("Creating dataset...")
    # start_time = time.time()
    # create_dataset(
    #     oracle,
    #     or_tokenizer,
    #     student,
    #     st_tokenizer,
    #     topics_list,
    #     professions,
    #     args.num_subtopic_per_topic,
    #     args.num_exercise_per_subtopic,
    #     args.dataset_path,
    #     args.oracle_max_length,
    #     args.oracle_temperature,
    #     args.student_max_length,
    #     args.student_temperature,
    # )
    # end_time = time.time()

    # execution_time = end_time - start_time
    # print(f"Execution Time: {execution_time} seconds")


if __name__ == "__main__":
    main()