
import io
import os
from urllib.parse import urlparse
from http import HTTPStatus

import boto3
from elasticsearch import Elasticsearch
from smart_open import open as smart_open


DATABASE_URL = os.environ['DATABASE_URL']
DATABASE_INDEX = os.environ['DATABASE_INDEX']
DATABASE_USER = os.environ['DATABASE_USER']
DATABASE_PASS = os.environ['DATABASE_PASS']

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_KEY = os.environ['MODEL_KEY']

QUERY_FIELDS = ['count', 'songs']


def knn_search(es, embedding, n=5):
    query = {
        'field': 'embedding',
        'query_vector': embedding,
        'k': n,
        'num_candidates': 100
    }
    resp = es.knn_search(
        index=DATABASE_INDEX,
        knn=query
    )

    close_songs = [{
        'score': hit['_score'],
        'song': hit['_source']
    } for hit in resp['hits']['hits']]

    return close_songs


def lambda_handler(event, context):
    
    # Initial validation on shape of event
    # Make sure all query fields are present
    for field in QUERY_FIELDS:
        if field not in event['params']:
            return {'statusCode': HTTPStatus.BAD_REQUEST.value}
    
    song_count = event['params']['count']

    es = Elasticsearch(
        DATABASE_URL,
        http_auth=(DATABASE_USER, DATABASE_PASS)
    )

    # Load torch model from s3
    load_path = f's3://{MODEL_BUCKET}/{MODEL_KEY}'
    with smart_open(load_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(buffer))

    song_embeddings = dict()

    # Check which songs already exist in elasticsearch, if not compute new embedding
    for song in event['params']['songs']:
        song_id = song['id']
        resp = es.get(index=DATABASE_INDEX, id=song_id)
        if resp['found']:
            song_embeddings[song_id] = resp['_source']['embedding']
        else:
            song_embeddings[song_id] = model.predict(song['fields'])
    
    # Do knn search on elasticsearch for each song embedding
    per_song_recommendations = dict()
    for song_id in song_embeddings:
        per_song_recommendations[song_id] = knn_search(es, song_embeddings[song_id], n=song_count)

    # Merge per song recommendations into a list of 'song_count' length that contains score per original input song

    return {
        'statusCode': HTTPStatus.OK.value
    }