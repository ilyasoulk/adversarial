import json

def get_data(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print("Error: Unable to decode JSON")
            return None
    return data

def add_messages_attribute(data):
    for item in data:
        if 'prompt' in item and 'chosen' in item:
            item['messages'] = [
                {"content": item['prompt'], "role": "user"},
                {"content": item['chosen'], "role": "assistant"}
            ]
    return data

def save_processed_data(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Processed data saved to {output_file}")


# file_path = "datasets/dataset_mistral7b_phi.json"
file_path = "datasets/Qwen2.5-3B-Instruct_v0.json"
# Main process
data = get_data(file_path)
print(len(data))
# if data is not None:
#     processed_data = add_messages_attribute(data)
#     save_processed_data(processed_data, "datasets/dataset_mistral7b_phi_sft.json")
# else:
#     print("Failed to process data due to JSON decoding error.")