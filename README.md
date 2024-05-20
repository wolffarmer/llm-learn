# llm-learn Note
just learing llama llm model steps by steps

# Step1: run and learning llama inference code on pc platform（5.20-6.5）

## 1.1 clone llama code from hugging face(TinyLlama/TinyLlama-1.1B-Chat-v1.0)

using domestic mirror address in china(https://hf-mirror.com/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for example

```shell
git clone https://hf-mirror.com/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 1.2 install ide env(python,pycharm),pip install pkg

```shell
pip install transformers==4.35 numpy pandas torch
```

## 1.3 run script

```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://hf-mirror.com/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```

## 1.4 python debug mode to watch all varients step by step

![image](https://github.com/wolffarmer/llm-learn/assets/15634187/7a315de7-cb64-4e8c-ae2c-c12f404c205b)


# Step2: run llama code in orangepi ascend platform（6.6-6.20）-- TODO

# Step3: train llama model in pc platform(6.21-7.21) -- TODO

# Step4: Modify model training using data parallelism and model parallelism using multiple GPUs(7.21-8.21) -- TODO

