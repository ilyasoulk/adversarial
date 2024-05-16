import json
import os

dir_name = 'exercises_v2'
filename = 'exercises.json'

dir_path = os.path.join(os.getcwd(), dir_name)
# Iterate over the directory and exercises and read the JSON files to create a single JSON file
exercises = []
print('Starting to generate the exercises')
for subdir, _, files in os.walk(dir_path):
    print(subdir)
    for file in files:
        print(file)
        if file.endswith('.json'):
            with open(os.path.join(subdir, file), 'r') as f:
                exercises.extend(json.load(f))


# Write the exercises to a single JSON file
with open(filename , 'w') as f:
    json.dump(exercises, f, indent=4)