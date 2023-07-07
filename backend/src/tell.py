
import os
from urllib.parse import urlparse
from http import HTTPStatus


DATABASE_URL = os.environ['DATABASE_URL']


def lambda_handler(event, context):
    
    print('Hello world')

    return {
        'statusCode': HTTPStatus.OK.value
    }