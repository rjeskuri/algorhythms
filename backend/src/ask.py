
import io
import os
from urllib.parse import urlparse
from http import HTTPStatus

import boto3
from elasticsearch import Elasticsearch
from smart_open import open as smart_open
#import torch
import numpy as np


DATABASE_URL = os.environ['DATABASE_URL']
DATABASE_INDEX = os.environ['DATABASE_INDEX']
DATABASE_USER = os.environ['DATABASE_USER']
DATABASE_PASS = os.environ['DATABASE_PASS']

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_KEY = os.environ['MODEL_KEY']

QUERY_FIELDS = ['count', 'songs']


def knn_search(es, song_id, embedding, n=5):
    query = {
        'field': 'embedding',
        'query_vector': embedding,
        'k': n,
        'num_candidates': 1000
    }
    resp = es.knn_search(
        index=DATABASE_INDEX,
        knn=query
    )

    close_songs = [{
        'score': hit['_score'],
        'song': hit['_source'],
        'id': hit['_id']
    } for hit in resp['hits']['hits'] if hit['_id'] != song_id]

    return close_songs


def calculate_recommendations(count, per_song_recommendations):

    # Invert the dictionary so that we have recommended songs, with data about which songs informed that recommendation
    inverted_recommendations = dict()
    for source_song_id, single_song_recommendations in per_song_recommendations.items():
        for recommended_song in single_song_recommendations:
            id_ = recommended_song['id']
            score = recommended_song['score']
            if id_ not in inverted_recommendations:
                inverted_recommendations[id_] = {
                    'id': id_,
                    'name': 'test_name',
                    'sources': dict()
                }
            inverted_recommendations[id_]['sources'][source_song_id] = {
                'score': score,
                'id': source_song_id
            }

    # Compute overall recommendation score for each recommended song
    # Recommendation score = average per song recommendation score / log(1 + count of songs which recommended this song)
    for recommended_song_id, recommended_song in inverted_recommendations.items():
        average_score = np.mean([recommended_song['sources'][source_song_id]['score'] for source_song_id in recommended_song['sources']])
        recommendation_weight = np.log(1 + len(recommended_song['sources']))
        inverted_recommendations[recommended_song_id]['score'] = average_score * recommendation_weight
        for source_song_id, single_song_recommendation in recommended_song['sources'].items():
            recommended_song['sources'][source_song_id] = {
                **single_song_recommendation,
                'score': single_song_recommendation['score'] * recommendation_weight
            }

    # Sort all recommended songs to select only 'count' highest scores
    # Normalize all scores so that lowest (highest) score is 1
    recommendations = [song for song in sorted(inverted_recommendations.values(), key=lambda song: song['score'])][:count]
    recommendations = {song['id']: {
        **song,
        'score': song['score']
    } for song in recommendations}

    return recommendations


def lambda_handler(event, context):
    
    # Initial validation on shape of event
    # Make sure all query fields are present
    for field in QUERY_FIELDS:
        if field not in event:
            return {'statusCode': HTTPStatus.BAD_REQUEST.value}
    
    song_count = event['count']

    es = Elasticsearch(
        DATABASE_URL,
        http_auth=(DATABASE_USER, DATABASE_PASS)
    )

    """
    # Load torch model from s3
    load_path = f's3://{MODEL_BUCKET}/{MODEL_KEY}'
    with smart_open(load_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        model.load_state_dict(torch.load(buffer))
    """

    song_embeddings = dict()

    # Check which songs already exist in elasticsearch, if not compute new embedding
    for song in event['songs']:
        song_id = song['id']
        resp = es.get(index=DATABASE_INDEX, id=song_id)
        if resp['found']:
            song_embeddings[song_id] = resp['_source']['embedding']
        else:
            song_embeddings[song_id] = model.predict(song['fields'])
    
    # Do knn search on elasticsearch for each song embedding
    per_song_recommendations = dict()
    for song_id in song_embeddings:
        per_song_recommendations[song_id] = knn_search(es, song_id, song_embeddings[song_id], n=song_count)

    # Merge per song recommendations into a list of 'song_count' length that contains score per original input song
    recommendations = calculate_recommendations(song_count, per_song_recommendations)

    return {
        'statusCode': HTTPStatus.OK.value,
        'body': recommendations
    }