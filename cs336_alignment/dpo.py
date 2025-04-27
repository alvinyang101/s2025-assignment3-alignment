import torch
from transformers import PreTrainedTokenizerBase

def compute_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    
    formatted_chosen = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
                        f"\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_chosen}{tokenizer.eos_token}"
    )
    formatted_rejected = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
                        f"\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_rejected}{tokenizer.eos_token}"
    )
    
    # Tokenize
    chosen_tokens = tokenizer(formatted_chosen, return_tensors="pt")
    rejected_tokens = tokenizer(formatted_rejected, return_tensors="pt")
    
    prompt_only = f"### Instruction:\n{prompt}\n\n### Response:\n"
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt")
    prompt_len = prompt_tokens["input_ids"].shape[1]
    
    with torch.no_grad():
        chosen_output = lm(**chosen_tokens)
        rejected_output = lm(**rejected_tokens)
        
        # Get log probabilities for response tokens only
        chosen_logits = chosen_output.logits[:, prompt_len-1:-1, :]  # -1 to exclude EOS prediction
        rejected_logits = rejected_output.logits[:, prompt_len-1:-1, :]
        
        # Get the corresponding target token ids
        chosen_targets = chosen_tokens["input_ids"][:, prompt_len:]
        rejected_targets = rejected_tokens["input_ids"][:, prompt_len:]
        
        # Calculate log probabilities
        chosen_log_probs = gather_log_probabilities(chosen_logits, chosen_targets)
        rejected_log_probs = gather_log_probabilities(rejected_logits, rejected_targets)
        
        ref_chosen_output = lm_ref(**chosen_tokens) 
        ref_rejected_output = lm_ref(**rejected_tokens)
        
        ref_chosen_logits = ref_chosen_output.logits[:, prompt_len-1:-1, :]
        ref_rejected_logits = ref_rejected_output.logits[:, prompt_len-1:-1, :]
        
        ref_chosen_log_probs = gather_log_probabilities(ref_chosen_logits, chosen_targets)
        ref_rejected_log_probs = gather_log_probabilities(ref_rejected_logits, rejected_targets)
    
    chosen_policy_advantage = chosen_log_probs.sum() - ref_chosen_log_probs.sum()
    rejected_policy_advantage = rejected_log_probs.sum() - ref_rejected_log_probs.sum()
    
    policy_advantage_diff = chosen_policy_advantage - rejected_policy_advantage
    loss = -torch.nn.functional.logsigmoid(beta * policy_advantage_diff)
    
    return loss

def gather_log_probabilities(logits, targets):
    """
    Gather log probabilities for the target tokens from the logits.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        targets: Tensor of shape (batch_size, seq_len)
        
    Returns:
        Tensor of shape (batch_size, seq_len) containing log probabilities
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gathered_logits = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return gathered_logits