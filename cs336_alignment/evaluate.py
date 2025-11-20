from typing import List, Callable

from tqdm import tqdm
from vllm import LLM, SamplingParams

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    labels: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
        Evaluate a language model on a list of prompts,
        compute evaluation metrics, and serialize results to disk.
    """
    if len(prompts) != len(labels):
        raise ValueError("len(prompts) != len(labels)")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=True)
    for idx, (output, label) in enumerate(tqdm(zip(outputs, labels), total=len(outputs))):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward_dict = reward_fn(output, label)
        print(f"prompt: {prompt!r}, output: {generated_text!r}, label: {label!r},"
              f"reward: {reward_dict}")


