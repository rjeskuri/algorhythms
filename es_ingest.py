
import argparse
import os

from elasticsearch import Elasticsearch
import pandas as pd


DEFAULT_ELASTICSEARCH_URL = 'http://localhost:9200'
DEFAULT_ELASTICSEARCH_INDEX = 'song-embeddings'
DEFAULT_ELASTICSEARCH_USERNAME = 'user'
DEFAULT_ELASTICSEARCH_PASSWORD = 'password'

SONG_MAPPING = {
    'properties': {
        'song_id': {'type': 'text'},
        'name': {'type': 'text',}
        'embedding': {'type': 'double'}
    }
}


def init_elasticsearch(index, url, username, password):
    """
    Returns elasticsearch instance and creates index (database).
    """
    es = Elasticsearch(
        url,
        http_auth=(username, password),
        verify_certs=False
    )

    resp = es.options(ignore_status=[400]).indices.create(
        index=index,
        mapping=SONG_MAPPING
    )
    print(resp)

    return es


def load_song_embeddings(path):
    """
    Loads CSV of song embeddings and converts into a list of dictionaries.
    """
    song_df = pd.read_csv(path)

    songs = [{
        'song_id': row['song_id'],
        'name': row['name'],
        'embedding': row['embedding']
    } for _, row in song_df.iterrows()]

    return songs


def index_songs(es, index, songs):
    """
    Inserts songs into elasticsearch index.
    """
    for song in songs:
        song_id = song['song_id']
        es.index(
            index=index,
            document=song,
            id=song_id
        )


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='path to csv containing songs and embeddings')
    parser.add_argument('--es-url', help='url pointing to elasticsearch', default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument('--index', help='name of index to place embeddings into', default=DEFAULT_ELASTICSEARCH_INDEX)
    parser.add_argument('--username', help='elasticsearch credentials', default=DEFAULT_ELASTICSEARCH_USERNAME)
    parser.add_argument('--password', help='elasticsearch credentials', default=DEFAULT_ELASTICSEARCH_PASSWORD)

    args = parser.parse_args()

    es = init_elasticsearch(
        index=args.index,
        url=args.es_url,
        username=args.username,
        password=args.password
    )
    songs = load_song_embeddings(args.filename)

    index_songs(
        es=es,
        index=args.index,
        songs=songs
    )


if __name__ == '__main__':
    main()