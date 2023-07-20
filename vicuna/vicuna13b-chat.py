#!/usr/bin/env python3
import argparse

from rich.console import Console
from rich.markdown import Markdown

import cmd2

from fastchat import client


PROMPT_TEMPLATE = """
You are a friendly AI assistant chatbot that helps people with their questions. You have a friendly personality and are able to answer questions about the world around you.

USER INPUT: {prompt}'
"""


class VicunaChat(cmd2.Cmd):
    prompt = "vicuna> "

    def __init__(self, model_name: str):
        super().__init__(
            allow_cli_args=False,
            allow_redirection=False,
            shortcuts={},
        )
        self.console = Console()
        self.model_name = model_name
        self.client = client

        with self.console.capture() as capture:
            self.print(f'[bold red]{self.prompt}[/]', end='')
        self.prompt = capture.get()


    def generate(self, input_text: str):
        answer = None
        prompt = PROMPT_TEMPLATE.format(prompt=input_text)

        try:
            completion = self.client.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
               ]
            )
            answer = completion.choices[0].message.content

        except Exception as err:
            print(f'[error] caught exception making vicuna api request - {err}', 'error')
            return None

        return answer


    def default(self, statement: cmd2.Statement):
        self.process(statement.raw)


    def process(self, input_text: str):
        if not input_text:
            return

        answer = self.generate(input_text)
        self.console.print(Markdown(answer))
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '-n', '--name', 
        help='model name', 
        default='vicuna_13b'
        required=True
    )
    
    args = parser.parse_args()

    app = VicunaChat(args.name)
    app.cmdloop()
