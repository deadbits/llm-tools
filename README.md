# llm-tools
Collection of tools to assist with using Large Large Models (LLM)

## Overview 📖
The ability to run an LLM on your home computer is a huge resource for productivity and development. This repo contains a handful of one-off scripts and demos for interacting locally hosted LLMs, and some examples using the LangChain, EmbedChain, and LlamaIndex frameworks.

**Index**
* [OpenAI](/openai)
* [RedPajama](/redpajama/)
* [Llama2](/llama2/)
* [MPT-7B](/mpt-7b/)
* [Vicuna](/vicuna/)
* [EmbedChain](/embedchain/)

### ⭐ Featured: embedchain helper
[embedchain](https://github.com/embedchain/embedchain) makes it very easy to embed data, add it to a ChromaDB instance, and then ask questions about your data with an LLM. I created a small helpers to make this even easier: `ec-cli.py`

```
$ python ec-cli.py --help
usage: ec-cli.py [-h] [-e EMBED] [--text TEXT] [-q QUERY] [-m {openai,llama2}]

EmbedChain

options:
  -h, --help            show this help message and exit
  -e EMBED, --embed EMBED
                        add new resource to db
  --text TEXT           add text from local file
  -q QUERY, --query QUERY
                        Query the model
  -m {openai,llama2}, --model {openai,llama2}
                        llm model
```

![ec-cli.py demo](/assets/embedchain-cli.png)

Data added with the `--embed` or `--text` arguments is ingested into your ChromaDB.
You can also run [ec-api-server.py](/embedchain/ec-api-server.py) and posting to the `/embed` endpoint.

You can then query your data using the `--query` argument or the `/query` endpoint of the API server.

## Stack
Running models and tools locally is all good and well, but pretty quickly you'll want a more robust stack for things like:

* Inference hosting
* Orchestration
* Retrieving data from external sources
* Providing access to external tools
* [Managing prompts](https://github.com/deadbits/prompt-serve)
* Application hosting
* Interaction via common applications (iMessage, Telegram, etc.)
* Maintain memory/history of past interactions
* Embeddings model
* Store vector embeddings and metadata
* Manage documents prior to embeddings creation
* Logging

The list below includes a few of my favorites:
* [prompt-serve](https://github.com/deadbits/prompt-serve)
* [LlamaIndex](https://github.com/jerryjliu/llama_index)
* [embedchain](https://github.com/embedchain/embedchain)
* [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
* [ChromaDB](https://www.trychroma.com/)
* [FastChat](https://github.com/lm-sys/FastChat)
* [Gradio](https://www.gradio.app/)
* [OpenAI](https://openai.com/)
* [RedPajama](https://www.together.xyz/blog/redpajama-models-v1)
* [Mosaic ML](https://huggingface.co/mosaicml)
* [GPTCache](https://github.com/zilliztech/GPTCache)
* [Lambda Cloud](https://cloud.lambdalabs.com/)
* [Metal](https://getmetal.io/)
* [BentoML](https://github.com/ssheng/BentoChain)
* [Modal](https://modal.com/)
