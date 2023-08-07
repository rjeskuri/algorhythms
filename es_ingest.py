
import argparse
import os
import pickle

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd


DEFAULT_ELASTICSEARCH_URL = 'http://localhost:9200'
DEFAULT_ELASTICSEARCH_INDEX = 'song-embeddings'
DEFAULT_ELASTICSEARCH_USERNAME = 'user'
DEFAULT_ELASTICSEARCH_PASSWORD = 'password'

SONG_MAPPING = {
    'mappings': {
        'properties': {
            'song_id': {'type': 'text'},
            'track_uri': {'type': 'text'},
            'track_name': {'type': 'text'},
            'track_artist': {'type': 'text'},
            'embedding': {
                'type': 'dense_vector',
                'dims': 10,
                'index': True,
                'similarity': 'l2_norm'
            }
        }
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

    es.indices.delete(index=index, ignore=[400, 404])

    resp = es.indices.create(
        index=index,
        ignore=400,
        body=SONG_MAPPING
    )
    print(resp)

    return es


def load_song_embeddings(index_path, name_path, embedding_path):
    """
    Loads song embeddings and converts into a list of dictionaries.
    """
    with open(index_path, 'rb') as fp:
        index = pickle.load(fp)
    with open(name_path, 'rb') as fp:
        names = pickle.load(fp)
    with open(embedding_path, 'rb') as fp:
        embeddings = pickle.load(fp)
    embeddings = embeddings.cpu().detach().numpy()

    songs = [{
        'song_id': song_id,
        'track_uri': names[index][0],
        'track_name': names[index][1],
        'track_artist': names[index][2],
        'embedding': list(embeddings[index])
    } for song_id, index in index.items()]

    return songs


def index_songs(index, songs):
    """
    Inserts songs into elasticsearch index.
    """
    for song in songs:
        song_id = song['song_id']
        doc = {
            '_index': index,
            '_id': song_id,
            '_source': song
        }
        yield doc


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('index_filename', help='path to dictionary pkl containing songs ids')
    parser.add_argument('names_filename', help='path to pickle object containing song names')
    parser.add_argument('embedding_filename', help='path to dictionary pkl containing songs embeddings')
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
    songs = load_song_embeddings(args.index_filename, args.names_filename, args.embedding_filename)

    bulk(es, index_songs(
        index=args.index,
        songs=songs
    ))


if __name__ == '__main__':
    main()