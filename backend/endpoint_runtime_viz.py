
import random
import requests
import time
import json

from matplotlib import pyplot as plt

ENDPOINT_URL = ''
SONGS = {

}

def query_backend(query):

    start_time = time.time()

    resp = requests.post(url=ENDPOINT_URL, data=json.dumps(query))

    query_time = time.time() - start_time
    return query_time


def render_plot():
    pass


def main():
    pass
    


if __name__ == '__main__':
    main()