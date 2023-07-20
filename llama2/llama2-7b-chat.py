#!/usr/bin/env python3
import sys
import torch

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""
B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

class Model:
    def __init__(self):
        self.model = LlamaForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf'
        )

    def generate(self, prompt, stream=False, temperature=0.1, top_p=0.75, top_k=40, num_beams=1, max_length=512, **kwargs):
        args = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.2,
            max_length=max_length,
            **kwargs,
        )
        
        output = []
        full_prompt = f'{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}'
        inputs = self.tokenizer(full_prompt, return_tensors='pt', truncation=True, padding=False, max_length=1056)
        input_ids = inputs['input_ids'].to('cuda')
        
        with torch.no_grad():
            result = self.model.generate(
                input_ids=input_ids,
                generation_config=args,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                early_stopping=True,
            )

        for seq in result.sequences:
            output.append(self.tokenizer.decode(seq, skip_special_tokens=True).replace(full_prompt, ""))

        return output


if __name__ == '__main__':
    model = Model()

    while True:
        user_prompt = input('prompt> ')
        if user_prompt.lower() == '!quit' or user_prompt.lower() == '!q':
            print('goodbye!')
            break
    
        answer = model.generate(user_prompt)
        for ans in answer:
            print(ans)
    
    sys.exit(0)