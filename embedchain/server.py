#!/usr/bin/env python3
##
# embedchain API server
# add data to your embedchain via HTTP POST and query via HTTP GET
##
import os
import sys
import argparse

from rich import print as rprint

from embedchain import App
from embedchain import Llama2App

from flask import Flask, request, jsonify
from embedchain import App, Llama2App


app = Flask(__name__)


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


@app.route('/query', methods=['GET'])
def query():
    user_prompt = request.args.get('prompt', default = '', type = str)
    response = bot.query(user_prompt)
    return jsonify({'response': response})


@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()

    item_type = determine_type(data['item'])
    if item_type == 'text':
        bot.add_local('text', data['item'])
    else:
        bot.add(item_type, data['item'])

    return jsonify({'status': f'resource added to db: {data["item"]}'})


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()

    response = bot.summarize(data['item'])
    return jsonify({'response': response})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmbedChain')

    parser.add_argument(
        '-m', '--model',
        default='openai',
        type=str,
        choices=['openai', 'llama2'],
        help='llm model'
    )

    parser.add_argument(
        '-k', '--key',
        help='openai api key',
        required=False,
        action='store'
    )

    args = parser.parse_args()

    if args.model == 'openai':
        if not args.key:
            print('no openai api key provided; will attempt to use OPENAI_API_KEY environment variable')
        else:
            os.environ['OPENAI_API_KEY'] = args.key

        try:
            bot = App()
            rprint(f'[bold green]using openai model[/bold green]')
        except Exception as err:
            rprint(f'[bold red](error)[/bold red] failed to load openai model: {err}')
            sys.exit(1)

    elif args.model == 'llama2':
        try:
            bot = Llama2App()
            rprint(f'[bold green]using llama2 model via replicate[/bold green]')
        except Exception as err:
            rprint(f'[bold red](error)[/bold red] failed to load llama2 model: {err}')
            sys.exit(1)

    app.run()
