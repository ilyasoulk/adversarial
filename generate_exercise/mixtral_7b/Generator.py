import requests
import random
import re
import json
import os


class Generator:
    categories = []
    prompt_1 = '''
    I'm working on developing a comprehensive database of coding exercises for educational purposes and need your assistance in generating a list of the main types/categories of coding exercises. Note that these types should be broad enough to define sub-types for each of them. Each type must be between the ## and ## markers.
    Please ensure that the list is correctly formatted as I said previously to make it easier for me to parse. Generate only the types and not the sub-types.
    For example :
    ##Algorithm##
    ##Data Structure##
    Complete the list
    '''
    subcategories = []
    category = ''

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
            "top_p": 0.95,
        }

        response = requests.post(url, headers=headers, json=data)
        text = response.json()["choices"][0]["text"]
        return text

    def generate_categories(self):
        text = self.caller(self.prompt_1, 64)
        self.categories = self.parse(text, True)
        print('Categories:', self.categories)
        return self.categories

    def parse(self, text, first):
        matches = re.findall(r'##(.*?)##', text)
        if len(matches) == 0:
            print("No matches found")
            if first:
                return self.generate_categories()
            else:
                return self.generate_subtypes()
        return matches

    def select_random_category(self):
        self.category = random.choice(self.categories).rstrip()
        category_dir = os.path.join(os.getcwd(), 'exercises', self.category)
        os.makedirs(category_dir, exist_ok=True)

    def generate_subtypes(self):
        print(f'Selected category: {self.category}')
        prompt = f'''I need you to generate a list of type of programming exercise revolving around the category : {self.category}. 
        Each type must be between the ## and ## markers. Generate only the types and not the exercises.
        For example if the category is Data Structure, the subtypes could be ##Linked List##, ##Stack##, ##Queue##, etc.
        '''
        text = self.caller(prompt, 64)
        self.subcategories = self.parse(text, False)
        print('Subcategories:', self.subcategories)
        return self.subcategories

    def generate_exercises(self):
        output_dir = os.path.join(os.getcwd(), 'exercises')
        for subcategory in self.subcategories:
            subcategory = subcategory.rstrip()
            prompt = '''I need you to generate a JSON list of 3 coding exercises revolving around the subcategory :''' + subcategory + '''. 
            
            The JSON list should be in the following format:
            [JSON]
            [
                {
                    "title": "Exercise Title",
                    "description": "Description of the exercise",
                    "entry_point": "Function name to be implemented",
                }
                Complete the JSON
            ]
            [/JSON]
            Make sure the JSON is valid and correctly formatted.
            '''
            text = self.caller(prompt, 512)
            exercises = json.loads(text)
            filename = subcategory + '.json'
            category_dir = os.path.join(output_dir, self.category)
            with open(os.path.join(category_dir, filename), 'w') as file:
                json.dump(exercises, file, indent=4)
        return

gen = Generator()
gen.generate_categories()
gen.select_random_category()
gen.generate_subtypes()
gen.generate_exercises()