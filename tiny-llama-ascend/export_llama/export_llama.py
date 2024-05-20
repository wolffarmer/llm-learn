import os
import sys
from typing import List
import torch
import transformers
import torch_npu
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch_npu


def export_onnx(
    base_model: str = "models/tiny-llama",
    output_dir: str = "models/OriginOnnx/tiny-llama.onnx",
):
    device = torch.device('npu:0')
    tokenizer= LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    ).to(device)
    quantize_cfg = {
		"q_proj":{
			"type":"W8X8",
			"act_scale":False
		},"k_proj":{
			"type":"W8X8",
			"act_scale":False
		},"v_proj":{
			"type":"W8X8",
			"act_scale":False
		},"o_proj":{
			"type":"W8X8",
			"act_scale":False
		},"gate_proj":{
			"type":"W8X8",
			"act_scale":False
		},"up_proj":{
			"type":"W8X8",
			"act_scale":False
		}
	}
    input_names = ["input_ids", "attention_mask", "position_ids","past_key_values"]
    output_names = ["logits","out_key_values","attn_scores"]
    dynamic_axes = {
        "input_ids": { 0: "batch_size", 1: "seq_length" },
        "attention_mask": { 0: "batch_size",1:"all_len" },
        "position_ids": { 0: "batch_size", 1: "seq_length" },
        "past_key_values": { 2: "batch_size", 4: "kv_len" },
    }
    cfg=model.model.config
    batch_size,seq_len,kv_len=1,1,1024
    all_len = seq_len + kv_len
    n_layers,n_heads,hidden_size=cfg.num_hidden_layers,cfg.num_key_value_heads,cfg.hidden_size
    head_dim = int(cfg.hidden_size / cfg.num_attention_heads)


    input_ids = torch.zeros((batch_size,seq_len)).long().to("npu") # batch_size, new_sequence_length
    attention_mask = torch.zeros((batch_size,all_len)).long().to("npu") # batch_size, all_sequence_length
    position_ids = torch.zeros((batch_size,seq_len)).long().to("npu") # batch_size, new_sequence_length
    past_key_values = torch.rand((n_layers,2,batch_size,n_heads, kv_len, head_dim),dtype=torch.float16).to("npu")
    input_args = (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        None, # inputs_embeds: Optional[torch.FloatTensor] = None,
        None, #labels: Optional[torch.LongTensor] = None,
        True, #use_cache: Optional[bool] = None,
        True # output_attentions: Optional[bool] = None,
    )
    model.eval()
    with torch.no_grad():
        from quantize import quantize
        quantize(model,cfg=quantize_cfg)
        torch.onnx.export(
            model,
            f=output_dir,
            args=input_args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            export_params=True,
        )


if __name__ == "__main__":
    export_onnx()
