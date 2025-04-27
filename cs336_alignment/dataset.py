import gzip
import json
from typing import List, Dict

def load_anthropic_hh_dataset(file_paths: List[str]) -> List[Dict]:
    """
    Load the Anthropic HH dataset from a list of .jsonl.gz files,
    applying the necessary preprocessing.

    Args:
        file_paths (List[str]): List of paths to .jsonl.gz files.

    Returns:
        List[Dict]: A list of dictionaries, each with keys:
            - 'instruction' (str): the first human message
            - 'chosen' (str): assistant response preferred by human
            - 'rejected' (str): assistant response rejected by human
            - 'source' (str): filename the example came from
    """
    all_examples = []

    for file_path in file_paths:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                chosen = data["chosen"].strip()
                rejected = data["rejected"].strip()
                chosen_turns = chosen.split("\n\n")
                rejected_turns = rejected.split("\n\n")

                # Skip if more than 2 turns
                if len(chosen_turns) > 2 or len(rejected_turns) > 2:
                    continue

                try:
                    instruction = chosen_turns[0]
                    assert instruction.startswith("Human:")

                    chosen_response = chosen_turns[1]
                    rejected_response = rejected_turns[1]

                    chosen_response = chosen_response.replace("Assistant:", "", 1).strip()
                    rejected_response = rejected_response.replace("Assistant:", "", 1).strip()

                    # Add to dataset
                    all_examples.append({
                        "instruction": instruction.replace("Human:", "", 1).strip(),
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "source": file_path,
                    })
                except (IndexError, AssertionError):
                    continue

    return all_examples