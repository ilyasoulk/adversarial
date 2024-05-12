import requests
import random
import re
import json
import os
import ast


class Generator:
    categories = []
    prompt_1 = '''
    I'm working on developing a comprehensive database of interview coding exercises for educational purposes and need your assistance in generating a list of the main types/categories of coding exercises. Note that these types should be broad enough to define sub-types for each of them. Each type must be between the ## and ## markers.
    Please ensure that the list is correctly formatted as I said previously to make it easier for me to parse. Generate only the types and not the sub-types.
    For example :
    ##Algorithm##
    ##Data Structure##
    Complete the list
    '''
    subcategories = []
    category = ''
    exercises = {}

    def __init__(self):
        filename = 'Filtered_categories.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
                self.categories = data['categories']

    @staticmethod
    def caller(prompt, size):
        url = "http://node10.lrde.epita.fr:8000/v1/completions"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": prompt,
            "max_tokens": size,
            "top_p": 0.90,
            "temperature": 0.6,
        }

        response = requests.post(url, headers=headers, json=data)
        text = response.json()["choices"][0]["text"]
        return text

    def select_random_category(self):
        self.category = random.choice(self.categories)
        self.subcategories = self.category['subcategories']
        name = self.category['name']
        print(f'Selected category: {name}')
        category_dir = os.path.join(os.getcwd(), 'exercises_v2', name)
        os.makedirs(category_dir, exist_ok=True)

    #def generate_subtypes(self):
    #    print(f'Selected category: {self.category}')
    #    prompt = f'''I need you to generate a list of type of programming exercise revolving around the category : {self.category}. 
    #    Each type must be between the ## and ## markers. Generate only the types and not the exercises.
    #    For example if the category is Data Structure, the subtypes could be ##Linked List##, ##Stack##, ##Queue##, etc.
    #    '''
    #    text = self.caller(prompt, 256)
    #    self.subcategories = self.parse(text, False)
    #    print('Subcategories:', self.subcategories)
    #    return self.subcategories



    def extract_json(self, text):
        # Using regular expression to find text enclosed in triple backticks
        json_data = re.search(r'```json(.*?)```', text, re.DOTALL)
        if json_data:
            return json_data.group(1).strip()
        else:
            return None
        

    def parse_json(self, json_string):
        try:
            # Parse the JSON string into a Python object
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            # If an error occurs during parsing, print an error message
            print(f"Error parsing JSON: {e}")
            return None
    
    def extract_and_parse_json_list(self, mixed_output):
        # Extract the JSON string from the mixed output
        json_string = self.extract_json(mixed_output)
        if json_string:
            # Parse the JSON string into a Python object
            return self.parse_json(json_string)
        else:
            return None

    def generate_exercises(self):
        name = self.category['name']
        print(f'Generating exercises for category: {name}')
        for subcategory in self.subcategories:
            print(f'Generating exercises for subcategory: {subcategory}')
            subcategory = subcategory.rstrip()
            prompt = '''I need you to generate a JSON list of 3 coding exercises revolving around the category : ''' + name + ''' and precisely the following subcategory : ''' + subcategory + '''. 
            
            The JSON list should be in the following format:
            ```json
            [
                {
                    "title": "Exercise 1 Title",
                    "description": "Description of the exercise 1, containing the problem statement and constraints, the arguments and their types, and the return type. The description should be the docstring of the function to be implemented.",
                    "entry_point": "Only the name of the function to be implemented and nothing else for example : "mergeList",
                }
                ...
            ]
            ```
            The exercise should not be to broad so we can easily write unit tests for them.
            Making sure the JSON is correctly formatted and parsable is the number one priority.
            You should only generate the JSON list and not any text outside of the JSON.
            All 3 coding exercises should be in the same JSON list.
            '''
            text = self.caller(prompt, 1028)
            exercises = self.extract_and_parse_json_list(text)
            if exercises:
                self.exercises[subcategory] = exercises
        return

    def write_exercises(self):
        output_dir = os.path.join(os.getcwd(), 'exercises_v2')
        name = self.category['name']
        for subcategory, exercises in self.exercises.items():
            print(f'Writing exercises for subcategory: {subcategory}')
            filename = subcategory + '.json'
            category_dir = os.path.join(output_dir, name)
            with open(os.path.join(category_dir, filename), 'w') as file:
                json.dump(exercises, file, indent=4)
        return

    def parse_unit_tests(self, text):
        # Using regular expression to find text enclosed in triple backticks
        unit_tests = re.search(r'\[UNIT TESTS\](.*?)\[/UNIT TESTS\]', text, re.DOTALL)
        if unit_tests:
            return unit_tests.group(1).strip()
        else:
            return None


    def add_unit_tests(self):
        for subcategory, exercises in self.exercises.items():
            print(f'Adding unit tests to exercises for subcategory: {subcategory}')
            modified_exercises = []
            for exercise in exercises:
                prompt = f'''For the following coding exercise, I want you to add unit tests to the function to be implemented:
                    - The unit tests should test the function with different inputs and edge cases
                    - The unit tests should be written in the form of assertions
                    - At least 3 unit tests should be added and at most 5
                    - You should not use any external libraries to write the unit tests
            Generate the unit tests between the [UNIT TESTS] and [/UNIT TESTS] tags. For example: [UNIT TESTS]assert entry_point(given_input_1) == expected_output_1\nassert entry_point(given_input_2) == expected_output_2\nassert entry_point(given_input_3) == expected_output_3[/UNIT TESTS]
            All the unit test should be in the same block and the block should be between the [UNIT TESTS] and [/UNIT TESTS] tags. Make sure that the block ends with [/UNIT TESTS].'''
                print(f'Adding unit tests to exercise: {exercise["title"]}')
                exercice_information = f"- Title: {exercise['title']}\n- Description: {exercise['description']}\n- EntryPoint: {exercise['entry_point']}\n- Docstring: {exercise['docstring']}"
                prompt = prompt + exercice_information + '\n'
                text = self.caller(prompt, 512)
                unit_tests = self.parse_unit_tests(text)
                if unit_tests:
                    exercise['unit_tests'] = unit_tests
                    modified_exercises.append(exercise)
            self.exercises[subcategory] = modified_exercises
        return

    def parse_docstrings(self, text):
        # Using regular expression to find text enclosed in triple backticks
        docstring = re.search(r'###(.*?)###', text, re.DOTALL)
        if docstring:
            return docstring.group(1).strip()
        else:
            return None

    def add_docstring(self):
        for subcategory, exercises in self.exercises.items():
            print(f'Adding docstring to exercises for subcategory: {subcategory}')
            modified_exercises = []
            for exercise in exercises:
                prompt = f'''For the following coding exercise, I want you to add a docstring to the function to be implemented:
                    - The docstring should be a multiline string enclosed in triple quotes
                    - The docstring should describe the function and its parameters along with their types
                    - The docstring should also describe the return type of the function
            Generate the docstring between the ### and ### markers.
            For example:
            ###def entry_point(param1: type, param2 : type) -> type : \n"""DOCSTRING"""###
            Here is the exercise:
            '''
                print(f'Adding docstring to exercise: {exercise["title"]}')
                exercice_information = f"- Title: {exercise['title']}\n- Description: {exercise['description']}\n- EntryPoint: {exercise['entry_point']}"
                prompt = prompt + exercice_information + '\n'
                text = self.caller(prompt, 512)
                docstring = self.parse_docstrings(text)
                if docstring:
                    exercise['docstring'] = docstring
                    modified_exercises.append(exercise)
            self.exercises[subcategory] = modified_exercises
        return




if __name__ == '__main__':
    gen = Generator()
    gen.select_random_category()
    gen.generate_exercises()
    gen.add_docstring()
    gen.add_unit_tests()
    gen.write_exercises()
