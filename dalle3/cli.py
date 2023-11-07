import argparse
import logging

from openai import OpenAI
from loguru import logger


def generate(prompt):
    try:
        response = client.images.generate(
          model="dall-e-3",
          prompt=prompt,
          size="1024x1024",
          quality="standard",
          n=1,
        )

        logger.success(f'Image URL: {response.data[0].url}')
    except Exception as err:
        logger.error(err)
        raise err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create images with dall-e 3')

    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='OpenAI API Key'
    )

    parser.add_argument(
        '--text',
        type=str,
        help='Text prompt',
        required=True
    )

    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    generate(args.text)