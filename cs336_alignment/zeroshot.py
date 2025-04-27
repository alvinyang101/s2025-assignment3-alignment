import re
from typing import Any

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    output = model_output.strip().upper()
    option_letters = ["A", "B", "C", "D"]

    # Check if the model output directly matches a valid option
    for option in option_letters:
        if re.fullmatch(rf"({option})[\).]?", output):
            return option

    match = re.search(r"ANSWER:\s*\(?([A-D])\)?(\s|$)", output)
    if match and match.group(1) in option_letters:
        return match.group(1)

    match = re.search(r"\b([A-D])\b", output)
    if match and match.group(1) in option_letters:
        return match.group(1)
    
    match = re.search(r"\b(?:ANSWER\s+IS|IT'S|IS)\s+([A-D])\b", output)
    if match and match.group(1) in option_letters:
        return match.group(1)

    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # Find all numbers, including integers and decimals
    numbers = re.findall(r'\d+(?:\.\d+)?', model_output)
    
    if not numbers:
        return None
    
    return numbers[-1]