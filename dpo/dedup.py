import json
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer
import argparse


def count_tokens(texts, tokenizer):
    num_tokens = 0
    for text in texts:
        tokens = tokenizer(text)
        num_tokens += len(tokens)
    return num_tokens


def preprocess(text):
    return text.lower().strip()


def get_data(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    chosens = [preprocess(sample["chosen"]) for sample in data]
    rejecteds = [preprocess(sample["rejected"]) for sample in data]
    prompts = [preprocess(sample["prompt"]) for sample in data]

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


def create_shingles(text, k=3):
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def dedup(texts, threshold=0.5):
    # Initialize LSH index
    lsh = MinHashLSH(threshold=threshold)

    # Store original texts and their hashes
    minhashes = {}
    unique_texts = []
    duplicates = []

    for idx, text in enumerate(texts):
        # Create MinHash object
        m = MinHash()
        shingles = create_shingles(text)
        for s in shingles:
            m.update(s.encode("utf8"))

        # Check for near-duplicates
        result = lsh.query(m)

        if not result:  # No near-duplicates found
            lsh.insert(str(idx), m)
            unique_texts.append(text)
            minhashes[str(idx)] = m
        else:
            duplicates.append(text)

    return unique_texts, duplicates, len(duplicates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="Path for dataset")

    parser.add_argument("--tokenizer_id", type=str, help="Path for tokenizer")

    args = parser.parse_args()

    dataset = get_data(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    length = len(dataset["prompt"])
    num_prompts_tokens = count_tokens(dataset["prompt"], tokenizer)
    num_chosen_tokens = count_tokens(dataset["chosen"], tokenizer)
    num_rejected_tokens = count_tokens(dataset["rejected"], tokenizer)

    total = num_rejected_tokens + num_chosen_tokens + num_prompts_tokens
    print(
        f"Number of tokens for prompts : {num_prompts_tokens}\nNumber of tokens for chosen : {num_chosen_tokens}\nNumber of tokens for rejected : {num_rejected_tokens}\nTotal number of tokens {total}"
    )
    print(f"Total length : {length}")
    # prompts
    _, _, num_dupes = dedup(dataset["prompt"], threshold=0.8)
    print(f"Number of duplicates for prompts: {num_dupes}")

    # chosens
    _, _, num_dupes = dedup(dataset["chosen"], threshold=0.8)
    print(f"Number of duplicates for chosens: {num_dupes}")

    # rejecteds
    _, _, num_dupes = dedup(dataset["rejected"], threshold=0.8)
    print(f"Number of duplicates for rejecteds: {num_dupes}")
