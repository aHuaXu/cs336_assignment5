from typing import List, Dict
import torch
from torchtyping import TensorType
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy


def tokenize_prompt_and_output(
        prompt_strs: List[str],
        output_strs: List[str],
        tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings.
            Then the returned dictionary should have the following keys:

            - input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
              the tokenized prompt and output strings, with the final token sliced off.

            - labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
              shifted input ids, i.e., the input ids without the first token.

            - response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
              a mask on the response tokens in the labels.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("len(prompt_strs) != len(output_strs)")

    # tokenize
    prompt_tokens = tokenizer(prompt_strs, padding=False, truncation=False, return_tensors=None)
    output_tokens = tokenizer(output_strs, padding=False, truncation=False, return_tensors=None)

    batch_size = len(prompt_strs)
    # concatenated_ids = torch.cat([prompt_tokens["input_ids"], output_tokens["input_ids"]], dim=1)
    concatenated_ids = []
    for i in range(batch_size):
        concatenated_ids.append(prompt_tokens["input_ids"][i] + output_tokens["input_ids"][i])

    padded = tokenizer.pad(
        {"input_ids": concatenated_ids},
        padding=PaddingStrategy.LONGEST,
        return_tensors="pt",
    )
    padded_input_ids = padded["input_ids"]
    max_seq_len = padded_input_ids.shape[1]

    input_ids = padded_input_ids[:, :-1] # (batch_size, max_seq_len-1)
    labels = padded_input_ids[:, 1:] # (batch_size, max_seq_len-1)

    response_mask = torch.zeros_like(input_ids)
    for i in range(batch_size):
        prompt_len = len(prompt_tokens["input_ids"][i])
        output_end = prompt_len + len(output_tokens["input_ids"][i]) - 1
        response_mask[i, prompt_len-1:output_end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
            prediction.

    Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    """
    lsm = torch.logsumexp(logits, dim=-1, keepdim=True)

    log_probs = logits - lsm
    probs = torch.exp(log_probs)

    return -torch.sum(log_probs * probs, dim=-1)