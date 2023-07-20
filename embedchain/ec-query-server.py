#!/usr/bin/env python3
# embedchain query server
# very basic example of querying embedchain via Flask API endpoint
# original: https://github.com/embedchain/embedchain/blob/main/notebooks/embedchain-docs-site-example.ipynb

from flask import Flask, request, jsonify
from embedchain import App


app = Flask(__name__)
bot = App()


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data['question']
    response = bot.query(question)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run()