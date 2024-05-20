import numpy as np
import os
import sys

from sentencepiece import SentencePieceProcessor
from typing import Generator, List,Tuple
import gc
from transformers import LlamaTokenizer
from enum import Enum
from threading import Lock

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

from session import Session
from config import InferenceConfig
class LlamaInterface:
    def __init__(self,config:InferenceConfig) -> None:
        self.max_length = config.max_length
        # self.tokenizer=Tokenizer(config.tokenizer)
        self.tokenizer:LlamaTokenizer=LlamaTokenizer.from_pretrained(config.tokenizer)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.max_cache_size = config.max_cache_size
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}
        self.reset()
        self.lock = Lock()
        self.first=True
        self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5}
        print("init success")


    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        self.first = False
        input_ids = np.asarray(self.tokenizer.encode(prompt),dtype=np.int64).reshape(1,-1)
        logits = self.session.run(input_ids)[0]
        return self.sample_logits(logits[0][-1:],self.sampling_method,self.sampling_value,self.temperature),logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    def predict(self, text):
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""   
        if text == "":
            return    

        text = preprocess(text)

        input_ids = self.tokenizer.encode(text)
        if not self.first:
            input_ids = [29871,13,29966] + input_ids[2:]
        self.first = False
        input_ids = np.asarray(input_ids,dtype=np.int64).reshape(1,-1)
        ids_list = []
        count = 0
        for i in range(self.max_length):
            if self.session.run_times >= self.max_cache_size: 
                self.reset()
                break
            logits = self.session.run(input_ids)[0]
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)
            count += 1
            # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
            if input_ids[0] == self.tokenizer.eos_token_id:
                text_out = self.tokenizer.decode(ids_list)
                # text = self.tokenizer.decode(a)
                sys.stdout.write(f'{text_out}\n')
                sys.stdout.write(f'count: {count}\n')
                sys.stdout.flush()
                with self.lock:
                    self.state['message'],self.state['isEnd'] = text_out.strip(), True
                del logits
                gc.collect()
                break
            ids_list.append(input_ids[0][0])           
            text_out = self.tokenizer.decode(ids_list)
            stop_word = is_stop_word_or_prefix(text_out,["<|user|>","<|assistant|>"])
            
            if stop_word != "":
                with self.lock:
                    self.state['message'],self.state['isEnd'] = text_out[:-len(stop_word)].strip(),True
                self.session.rollback(self.stop_mp[stop_word])
                break
            else:
                with self.lock:
                    self.state['message'],self.state['isEnd'] = text_out.strip(), False
        return text_out

    def reset(self):
        self.first = True
        self.session.run_times = 0
        self.session.reset()
        self.generate_cache(self.prompt)


    def getState(self):
        with self.lock:
            return self.state.copy()

def preprocess(text:str) -> str:
    # 将输入转换为指定格式
    return f"<|user|>\n{text}</s>\n<|assistant|>"
    

def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return stop_word
    return ""
