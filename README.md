# CodeLLaMA-chat

CodeLLaMA-chat - The project aims to train a conversational model based on CodeLLaMA, providing a multi-turn dialogue version to assist with code development and troubleshooting.

# Model
- Original base model
  - codellama base 34b: https://huggingface.co/TheBloke/CodeLlama-34B-fp16
  - codellama python 34b: https://huggingface.co/TheBloke/CodeLlama-34B-Python-fp16
  - codellama instruct 34b: https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-fp16
  - codellama base 13b: https://huggingface.co/TheBloke/CodeLlama-13B-fp16
  - codellama python 13b: https://huggingface.co/TheBloke/CodeLlama-13B-Python-fp16
  - codellama instruct 13b: https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-fp16
  - codellama base 7b: https://huggingface.co/TheBloke/CodeLlama-7B-fp16
  - codellama python 7b: https://huggingface.co/TheBloke/CodeLlama-7B-Python-fp16
  - codellama instruct 7b: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-fp16

## Datasets
- ShareGPT-90K's computer 26k
  - Chinese version: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k/blob/main/sharegpt_jsonl/computer_zh_26k.jsonl
  - English version: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k/blob/main/sharegpt_jsonl/computer_en_26k.jsonl
  - ... (You can also translate them to other languages with GPT)
  - change-info-tool for this data: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k/blob/main/shareGPT/change_info.py

## Todo
- Train a multi-turn dialogue version.
- Code inference and API deployment.
- Integration with IDE plugins.
