![image](https://github.com/wolffarmer/llm-learn/assets/15634187/4e1b744b-bb3e-4b75-910b-bb8ddaaf039f)# llm-learn Note

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


# Step2: run llama code in orangepi ascend platform（6.6-6.20）

## 2.1 download firmware and os package of oriange ai pro from official website or from my baidu cloud storge share link

- official website （also baidu cloud storage link）

[Orange Pi AIpro Orange Pi官网-香橙派（Orange Pi）开发板,开源硬件,开源软件,开源芯片,电脑键盘](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro.html)

- my baidu cloud storge share link
- 
链接：https://pan.baidu.com/s/1CUJdWhZwLqf1pT0YIod56A?pwd=7vt4 
提取码：7vt4
2.2 install os(ubunto) from "orange ai pro\用户手册\OrangePi_AI_Pro_昇腾_用户手册_v0.3.1.pdf" 

2.3 install system(ubtunto) and run sample 
![image](https://github.com/wolffarmer/llm-learn/assets/15634187/6be3b4b9-fdff-4ee8-a893-9a93d74cd78d)
![image](https://github.com/wolffarmer/llm-learn/assets/15634187/0ab488c7-9497-40bd-890e-cdf139aa99ca)
![image](https://github.com/wolffarmer/llm-learn/assets/15634187/0e34cebb-4e2c-4575-a252-afd9616d60d5)


# Step3: train llama model in pc platform(6.21-7.21) -- TODO

# Step4: Modify model training using data parallelism and model parallelism using multiple GPUs(7.21-8.21) -- TODO
