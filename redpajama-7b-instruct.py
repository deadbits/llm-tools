#!/usr/bin/env python
import cmd2
import torch
import transformers

from rich.console import Console
from rich.markdown import Markdown

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def make_prompt(user_prompt):
    prompt_template = f'Q: {user_prompt}\nA:'
    return prompt_template

class LLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('togethercomputer/RedPajama-INCITE-7B-Instruct')
        model = AutoModelForCausalLM.from_pretrained('togethercomputer/RedPajama-INCITE-7B-Instrut', torch_dtype=torch.float16)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to('cuda:0')
        self.max_tokens = 128
        self.temperature = 0.7
        self.top_p = 0.7
        self.top_k = 50
        self.do_sample = True
        self.return_dict_in_generate = True

    def generate(self, user_prompt):
        prompt = make_prompt(user_prompt=user_prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_tokens, 
            do_sample=self.do_sample, 
            temperature=self.temperature, 
            top_p=self.top_p,
            top_k=self.top_k, 
            return_dict_in_generate=self.return_dict_in_generate
        )

        token = outputs.sequences[0, input_len:]
        output_str = self.tokenizer.decode(token)
        return output_str


class Chat(cmd2.Cmd):
    prompt = "user > "

    def __init__(self):
        super().__init__(
            allow_cli_args=False,
            allow_redirection=False,
            shortcuts={},
        )
        self.console = Console()

        self.llm_client = LLM()

        with self.console.capture() as capture:
            self.console.print(f'[bold yellow]{self.prompt}[/]', end='')
        self.prompt = capture.get()

    def default(self, statement: cmd2.Statement):
        self.process(statement.raw)

    def process(self, input_text: str):
        if not input_text:
            return

        answer = self.llm_client.generate(input_text)
        self.console.print(Markdown(answer))
   

if __name__ == '__main__':
    assert transformers.__version__ >= '4.25.1', f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

    app = Chat()
    app.cmdloop()

