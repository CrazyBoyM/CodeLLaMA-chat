# CodeLLaMA-chat

CodeLLaMA-chat - The project aims to train a conversational model based on CodeLLaMA, providing a multi-turn dialogue version to assist with code development and troubleshooting.

# Model
- base model
  - codellama base 34b: https://huggingface.co/TheBloke/CodeLlama-34B-fp16
  - codellama python 34b: https://huggingface.co/TheBloke/CodeLlama-34B-Python-fp16
  - codellama instruct 34b: https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-fp16
  - codellama base 13b: https://huggingface.co/TheBloke/CodeLlama-13B-fp16
  - codellama python 13b: https://huggingface.co/TheBloke/CodeLlama-13B-Python-fp16
  - codellama instruct 13b: https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-fp16
  - codellama base 7b: https://huggingface.co/TheBloke/CodeLlama-7B-fp16
  - codellama python 7b: https://huggingface.co/TheBloke/CodeLlama-7B-Python-fp16
  - codellama instruct 7b: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-fp16
- Chat model
  - Chinese Chat Version
    - based model: CodeLlama-13B-fp16, train dataset: computer_zh_26k
    - model link: https://huggingface.co/shareAI/CodeLLaMA-chat-13b-Chinese
  - English Chat Version
    - Come soon

## Datasets
- ShareGPT-90K's computer 26k
  - Chinese version: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k/blob/main/sharegpt_jsonl/computer_zh_26k.jsonl
  - English version: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k/blob/main/sharegpt_jsonl/computer_en_26k.jsonl
  - ... (You can also translate them to other languages with GPT)
  - change-info-tool for this data: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k/blob/main/shareGPT/change_info.py
  - train-llm-tool for this data: https://github.com/yangjianxin1/Firefly

## How to use
```
# from Firefly
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    model_name = 'shareAI/CodeLLaMA-chat-13b-Chinese' # set with your model path

    device = 'cuda'
    max_new_tokens = 500    # max tokens for each response
    history_max_len = 1000  # memory length 
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )


    history_token_ids = torch.tensor([[]], dtype=torch.long)

    user_input = input('User：')
    while True:
        input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
        user_input_ids = torch.concat([input_ids, eos_token_id], dim=1)
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        model_input_ids = history_token_ids[:, -history_max_len:].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids)
        print("Bot：" + response[0].strip().replace(tokenizer.eos_token, ""))
        user_input = input('User：')


if __name__ == '__main__':
    main()
```

## Todo
- Train a multi-turn dialogue version.
- Code inference and API deployment.
- Integration with IDE plugins.
