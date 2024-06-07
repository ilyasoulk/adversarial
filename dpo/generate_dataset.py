import argparse
import json
import argparse
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from typing import List
from pydantic import BaseModel


class Topic(BaseModel):
    topic: str


class Exercise(BaseModel):
    exercise: str
    topic: Topic


class Query(BaseModel):
    query: str
    topic: Topic


def extract_list_from_string(text):
    # Pattern to match a list enclosed in square brackets
    # This regex assumes no nested brackets for simplicity
    pattern = r"\[(.*?)\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Extract the content inside the brackets
        list_content = match.group(1)
        # Split by commas and strip whitespace and quotes
        list_items = [
            item.strip().strip('"').strip("'") for item in list_content.split(",")
        ]
        return list_items
    return []


def extract_assistant_content(messages):
    return extract_code_block(messages[-1]["content"])


def extract_code_block(text, language="python"):
    # Regex to find code blocks for a specified language, capturing content inside backticks
    pattern = rf"```{language}\s*\n(.*?)\n```"
    # Use re.DOTALL to allow dot (.) to match newlines
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Return the first match; you could also handle multiple matches if necessary
        return matches
    print(f"NO PYTHON MARKDOWN WAS FOUND FOR \n{text}")
    return None


def extract_code_parts(full_code):
    # Regex pattern to capture the function header, args, docstring, and python code
    pattern = r"(def\s+.+?\(.*?\):\s*\"\"\"(?:.|\n)*?\"\"\")((?:.|\n)*)"

    # Search the pattern in the provided full code string
    print(f"full_code :\n {full_code}")
    match = re.match(pattern, full_code)

    if match:
        # Extracting the two groups: function + docstring, and python code
        function_docstring = match.group(1)
        python_code = match.group(2)
        return function_docstring, python_code
    else:
        print(f"WE DIDN'T FOUND ANY FUNCTION DEFINITION FOR THIS : \n{full_code}")
        return None, None  # Return None if no match found


def extract_code_block_rejected(text, language="python"):
    pattern = rf"```{language}\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[0]
    else:
        print(f"NO PYTHON MARKDOWN WAS FOUND FOR \n{text}")
        return text


def generate_rejected_solutions(
    student_pipe, exercises, dataset, max_length, temperature
):
    for prompt, chosen in exercises:
        if prompt == None:
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        completion = student_pipe(
            messages,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        rejected = extract_code_block_rejected(
            completion[0]["generated_text"][-1]["content"]
        )
        print(f"Rejected solution : {rejected}")
        dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return dataset


def create_subtopic_query(topic: str, n: int) -> str:
    return f"""For a Python textbook give me {n} subtopics of {topic}, formatted as an unamed Python list. For example ```python[subtopic_1, subtopic_2 ... subtopic_n]``` 
    Just provide the titles and give no explanation.
    Format the result as Python list.
    """


def create_prompt_query(
    topic: Topic, profession: str, n: int, reference: str = None
) -> str:
    if reference:
        prefix = f"""
        Here are some reference exercises where our students are having a hard time :\n{reference}
        """

    query = f'''
            Create {n} DISTINCT code completion exercise about “{topic.topic}””.  
            Write it for a {profession}. 

            The exercise must be of the style: 

            ```
            def name(args):

            """Docstring explaining the exercise"""

            the solution of the exercise in Python
            ```            
            NO CLASSES

            MAKE IT VERY DIFFICULT
            '''
    query = "\n".join([m.lstrip() for m in query.strip().split("\n")])
    return query


def create_subtopics(oracle, topic: Topic, n: int, retries: int = 10):
    success = False
    query = create_subtopic_query(topic.topic, n)
    result = []
    print(query)
    for i in range(retries):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
            completion = oracle(
                messages,
                max_length=1000,
                temperature=1.5,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
            )
            response = extract_assistant_content(completion[0]["generated_text"])
            subcategories = extract_list_from_string(response[0])
            if subcategories == None:
                print(
                    f"Generation failed for prompt, retrying {i + 1}/{retries}, error: {e}"
                )
                continue
            result = [Topic(topic=i) for i in subcategories]
            success = True
        except Exception as e:
            print(
                f"Generation failed for prompt, retrying {i + 1}/{retries}, error: {e}"
            )
        else:
            break

    if success:
        return result
    else:
        return []


def create_prompt(topic: Topic, professions: List[str], n: int) -> Query:

    profession = professions[np.random.randint(0, len(professions))]
    query = create_prompt_query(topic, profession, n)

    return query


def create_dataset(
    oracle,
    student,
    topics,
    professions,
    num_subtopics,
    num_exercises,
    dataset_path,
    oracle_max_length,
    oracle_temperature,
    student_max_length,
    student_temperature,
):
    subtopics_list = []
    for topic in topics:
        subtopics_list.extend(
            create_subtopics(oracle, Topic(topic=topic), num_subtopics)
        )

    queries = [
        create_prompt(subtopic, professions, num_exercises)
        for subtopic in subtopics_list
    ]
    dataset = []
    for query in tqdm(queries, ncols=100, desc="Generating dataset"):
        messages = [{"role": "user", "content": query}]
        completion = oracle(
            messages,
            max_length=oracle_max_length,
            temperature=oracle_temperature,
            num_return_sequences=1,
            top_k=50,
            top_p=0.95,
        )
        response = completion[0]["generated_text"]
        codes = extract_assistant_content(response)
        if codes:
            parsed_code = [extract_code_parts(code) for code in codes]
            dataset = generate_rejected_solutions(
                student, parsed_code, dataset, student_max_length, student_temperature
            )

    with open(dataset_path, "w") as f:
        json.dump(dataset, f)

    return dataset


def create_pipeline(model_path, device):
    return pipeline(
        "text-generation", model=model_path, tokenizer=model_path, device=device
    )


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


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
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = create_pipeline(args.oracle_path, device)
    student = create_pipeline(args.student_path, device)
    professions = load_json("tree/professions.json")
    topics_list = load_json("tree/topics.json")

    create_dataset(
        oracle,
        student,
        topics_list,
        professions,
        args.num_subtopic_per_topic,
        args.num_exercise_per_subtopic,
        args.dataset_path,
        args.oracle_max_length,
        args.oracle_temperature,
        args.student_max_length,
        args.student_temperature,
    )


if __name__ == "__main__":
    main()
