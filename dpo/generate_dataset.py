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
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
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


def create_prompt_query(
    topic: Topic, profession: str, n: int, exercise_type="code_completion"
) -> str:
    if exercise_type == "code_completion":
        return f'''
            Create {n} unique and challenging Python code completion exercises on the topic of "{topic}".
            Each exercise should be framed in the context of a {profession}'s work to make it more engaging.

            Structure each exercise as follows:
            
            ```
            def function_name(parameters):
                """
                Description of the task or problem to solve, providing all necessary context.
                """
                # Solution code starts here (in Python)
            ```
            
            Guidelines:
            - Avoid using classes
            - Make exercises challenging
            - Ensure a complete solution is provided
            '''
    else:  # natural_language
        return f"""
            Create {n} unique and challenging Python programming exercises on the topic of "{topic}",
            framed within the context of a {profession}'s daily work scenarios.

            Structure each exercise as follows:

            [PROBLEM]
            Problem: [Natural language description that presents the programming challenge in a {profession}-related scenario]

            Input: [Clearly specify the input format using domain-specific examples]

            Output: [Clearly specify the expected output format]

            Examples:
            [Provide 2-3 test cases with profession-relevant data and explanations]

            Constraints:
            [State any constraints or special considerations]
            [PROBLEM]

            Example scenario structure:
            - For a chef: "Given a list of recipe preparation times, find the optimal cooking schedule..."
            - For an architect: "Given dimensions of building materials, calculate the most efficient arrangement..."

            Guidelines:
            - Use profession-specific terminology and realistic scenarios
            - Make the problem mathematically sound while maintaining professional context
            - Include domain-relevant example data in test cases
            - Do NOT provide the solution, only the problem description
            """


def create_prompts(topic: Topic, professions: List[str], n: int) -> List[Query]:
    """Generate both code completion and natural language prompts for each topic"""
    profession = professions[np.random.randint(0, len(professions))]
    return [
        create_prompt_query(topic, profession, n, "code_completion"),
        create_prompt_query(topic, profession, n, "natural_language"),
    ]


def create_topic_prompts(topics, n, reference_exercises=[]):
    res = []
    if len(reference_exercises) > 0:  # Adversarial episode
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
            res.extend(
                [
                    incremental_complexity_prompt,
                    opposite_approach_prompt,
                    deceptive_complexity_prompt,
                    conceptual_shift_prompt,
                    error_based_prompt,
                ]
            )

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


def generate(
    model,
    tokenizer,
    batch_size,
    prompts,
    generation_config,
    desc="Generating...",
    instruct=False,
):
    responses = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i : i + batch_size]

        if instruct:
            inputs = tokenizer.apply_chat_template(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_dict=True,
            )
        else:
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )

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
                pad_token_id=tokenizer.eos_token_id,
            )

        outputs = outputs[:, inputs["input_ids"].shape[1] :]

        # Decode
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)

        print(f"Processed {i + len(batch)}/{len(prompts)} prompts")

    return responses


def extract_functions_code_completion(text):
    """
    Extracts function definitions with their implementations from Python code for code completion exercises.
    """
    pattern = r'(def\s+(\w+)\([^)]*\):)\s*("""(?:(?!""")[\s\S])*?""")\s*((?:(?!def\s+\w+\([^)]*\):)[\s\S])*?)(?=\s*def\s+\w+\([^)]*\):|\s*$)'
    matches = list(re.finditer(pattern, text))

    functions = []
    for match in matches:
        func_def = match.group(1)
        docstring = match.group(3)
        implementation = match.group(4).strip()

        implementation = re.sub(r"^\s*\n", "", implementation)
        implementation = re.sub(r"\n\s*$", "", implementation)

        lines = implementation.split("\n")
        if lines:
            indentations = [
                len(line) - len(line.lstrip()) for line in lines if line.strip()
            ]
            if indentations:
                min_indent = min(indentations)
                implementation = "\n".join(
                    line[min_indent:] if line.strip() else "" for line in lines
                )

        functions.append(
            {
                "func_def": f"{func_def}\n    {docstring}",
                "code": implementation,
                "type": "code_completion",
            }
        )

    return functions


def extract_natural_language_problems(text):
    """
    Extracts natural language programming problems from the generated text.
    """
    pattern = r"\[PROBLEM\](.*?)\[PROBLEM\]"
    matches = re.findall(pattern, text, re.DOTALL)

    problems = []
    for match in matches:
        problem = match.strip()
        problems.append({"problem": problem, "type": "natural_language"})

    return problems


def create_prompts_dataset(
    oracle,
    or_tokenizer,
    topics,
    num_subtopics,
    reference_exercises,
    professions,
    num_exercises,
    batch_size,
    generation_config,
):
    subtopics_list = []
    prompts = create_topic_prompts(
        topics, num_subtopics, reference_exercises=reference_exercises
    )
    if len(reference_exercises) == 0:
        topics_raw = generate(
            oracle,
            or_tokenizer,
            batch_size=batch_size,
            prompts=prompts,
            generation_config=generation_config,
            desc="Generating Topics...",
        )
        for topic_raw in topics_raw:
            subtopics_list.extend(extract_list_from_string(topic_raw))

        with open("tree/subtopics.json", "w") as f:
            json.dump(subtopics_list, f, indent=4)

        # Create prompts
        queries = []
        for subtopic in subtopics_list:
            queries.extend(create_prompts(subtopic, professions, num_exercises))

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
    queries = create_prompts_dataset(
        oracle,
        or_tokenizer,
        topics,
        num_subtopics,
        reference_exercises,
        professions,
        num_exercises,
        batch_size,
        generation_config_oracle,
    )

    # Generate both types of exercises for each query
    messages_dataset = []
    for query in queries:
        for exercise_type in ["code_completion", "natural_language"]:
            messages_dataset.append(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful coding assistant.",
                    },
                    {
                        "role": "user",
                        "content": create_prompt_query(query, exercise_type),
                    },
                ]
            )

    # Generate responses
    responses = generate(
        oracle,
        or_tokenizer,
        batch_size,
        messages_dataset,
        generation_config=generation_config_oracle,
        desc="Generating DPO Prompts",
        instruct=True,
    )

    # Extract exercises
    dataset = []
    for i, response in enumerate(responses):
        if i % 2 == 0:  # Code completion exercises
            dataset.extend(extract_functions_code_completion(response))
        else:  # Natural language exercises
            dataset.extend(extract_natural_language_problems(response))

    print(
        f"Extracted {len(dataset)} exercises ({sum(1 for d in dataset if d['type'] == 'natural_language')} natural language, {sum(1 for d in dataset if d['type'] == 'code_completion')} code completion)"
    )

    # Generate rejecteds only for code completion exercises
    dpo_prompts = [
        f"Complete the following python code :\n{data['func_def']}"
        for data in dataset
        if data["type"] == "code_completion"
    ]
    winning = generate(
        oracle,
        or_tokenizer,
        batch_size,
        dpo_prompts,
        generation_config=generation_config_oracle,
        desc="Generating Winning Solutions",
    )
    rejected = generate(
        student,
        st_tokenizer,
        batch_size,
        dpo_prompts,
        generation_config=generation_config_student,
        desc="Generating Rejected Solutions",
    )

    clean_dataset = []
    # Clean up and compose dataset
    code_completion_index = 0
    for data in dataset:
        if data["type"] == "code_completion":
            win = winning[code_completion_index]
            reject = rejected[code_completion_index]
            if data["func_def"] != "" and win != "" and reject != "":
                clean_dataset.append(
                    {
                        "prompt": data["func_def"],
                        "chosen": win,
                        "rejected": reject,
                        "type": "code_completion",
                    }
                )
            code_completion_index += 1
        else:
            clean_dataset.append(
                {"prompt": data["problem"], "type": "natural_language"}
            )

    with open(dataset_path, "w") as f:
        json.dump(clean_dataset, f, indent=4)

    return clean_dataset


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
        quantization_config=quantization_config if do_quantization else None,
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
    generation_config_oracle = {
        "temperature": args.oracle_temperature,
        "max_length": args.oracle_max_length,
    }
    generation_config_student = {
        "temperature": args.student_temperature,
        "max_length": args.student_max_length,
    }

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
