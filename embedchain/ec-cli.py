#!/usr/bin/env python3
# embedchain helper
# author: adam swanda
# repo: github.com/deadbits/llm-utils

import os
import sys
import argparse

from rich import print as rprint
from rich.markdown import Markdown
from string import Template

from embedchain import App
from embedchain import Llama2App
from embedchain.config import QueryConfig


def determine_type(item):
    if item.lower().endswith('.docx'):
        rprint(f'[bold orange3]detected docx file:[/bold orange] {item}')
        return 'docx_file'

    elif item.startswith('https://') or item.startswith('http://'):
        if item.startswith('https://www.youtube.com/watch?v='):
            rprint(f'[bold orange3]detected youtube video:[/bold orange3] {item}')
            return 'youtube_video'
        elif item.endswith('sitemap.xml'):
            rprint(f'[bold orange3]detectedsitemap:[/bold orange3] {item}')
            return 'sitemap'
        elif item.endswith('.pdf'):
            rprint(f'[bold orange3]detected pdf file:[/bold orange3] {item}')
            return 'pdf_file'
        else:
            rprint(f'[bold orange3]detected web page:[/bold orange3] {item}')
            return 'web_page'

    if item.lower().endswith('.pdf'):
        rprint(f'[bold orange3]detected pdf file:[/bold orange3] {item}')
        return 'pdf_file'
    else:
        rprint(f'[bold orange3]detected text:[/bold orange3] {item}')
        return 'text'


def embed(item=None, text_file=None):
    if text_file:
        rprint(f'[bold green]adding text from file:[/bold green] {text_file}')
        with open(text_file, 'r') as fp:
            item = fp.read()
        item_type = 'text'
    else:
        item_type = determine_type(item)

    if item_type == 'text':
        app.add_local('text', item)
    else:
        app.add(item_type, item)


def query(user_prompt):
    return app.query(user_prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmbedChain')

    parser.add_argument(
        '-e', '--embed',
        action='store',
        help='add new resource to db'
    )
    
    parser.add_argument(
        '--text', 
        action='store',
        help='add text from local file'
    )

    parser.add_argument(
        '-q', '--query',
        action='store',
        help='Query the model'
    )

    parser.add_argument(
        '-m', '--model',
        default='openai',
        type=str,
        choices=['openai', 'llama2'],
        help='llm model'
    )

    args = parser.parse_args()

    if args.model == 'openai':
        app = App()
        os.environ['OPENAI_API_KEY'] = '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'
        rprint(f'[bold green]using openai model[/bold green]')
    elif args.model == 'llama2':
        app = Llama2App()
        os.environ['REPLICATE_API_TOKEN'] = "xxx"
        rprint(f'[bold green]using llama2 model via replicate[/bold green]')

    if args.embed:
        rprint(f'[bold green]adding resource:[/bold green] {args.embed}')
        embed(args.embed)
    
    if args.text:
        rprint(f'[bold green]adding text from file:[/bold green] {args.text}')
        embed(None, args.text)

    if args.query:
        rprint(f'[bold green]querying:[/bold green] {args.query}')
        result = query(args.query)
        rprint('[bold white]result:[/bold white]')
        print(result)