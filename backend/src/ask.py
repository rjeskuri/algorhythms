
import os
from http import HTTPStatus
import json
import joblib
import pickle

import boto3
from elasticsearch import Elasticsearch
import numpy as np


DATABASE_URL = os.environ['DATABASE_URL']
DATABASE_INDEX = os.environ['DATABASE_INDEX']
DATABASE_USER = os.environ['DATABASE_USER']
DATABASE_PASS = os.environ['DATABASE_PASS']

MODEL_BUCKET = os.environ['MODEL_BUCKET']
OHE_KEY = os.environ['OHE_KEY']
SCALER_KEY = os.environ['SCALER_KEY']
WEIGHTS1_KEY = os.environ['WEIGHTS1_KEY']
WEIGHTS2_KEY = os.environ['WEIGHTS2_KEY']
BIAS1_KEY = os.environ['BIAS1_KEY']
BIAS2_KEY = os.environ['BIAS2_KEY']

QUERY_FIELDS = ['count', 'songs']

s3 = boto3.resource('s3')


def load_encoders():
    s3.Bucket(MODEL_BUCKET).download_file(OHE_KEY, '/tmp/ohe.joblib')
    s3.Bucket(MODEL_BUCKET).download_file(SCALER_KEY, '/tmp/scaler.joblib')
    
    oheObj = joblib.load('/tmp/ohe.joblib')
    minMaxScalerObj = joblib.load('/tmp/scaler.joblib')
    
    return oheObj, minMaxScalerObj


def retrieve_model():
    s3.Bucket(MODEL_BUCKET).download_file(WEIGHTS1_KEY, '/tmp/mlp_weight1.pkl')
    s3.Bucket(MODEL_BUCKET).download_file(WEIGHTS2_KEY, '/tmp/mlp_weight2.pkl')
    s3.Bucket(MODEL_BUCKET).download_file(BIAS1_KEY, '/tmp/mlp_bias1.pkl')
    s3.Bucket(MODEL_BUCKET).download_file(BIAS2_KEY, '/tmp/mlp_bias2.pkl')
    
    with open('/tmp/mlp_weight1.pkl', 'rb') as fp:
        weights1 = pickle.load(fp)
    with open('/tmp/mlp_weight2.pkl', 'rb') as fp:
        weights2 = pickle.load(fp)
    with open('/tmp/mlp_bias1.pkl', 'rb') as fp:
        bias1 = pickle.load(fp)
    with open('/tmp/mlp_bias2.pkl', 'rb') as fp:
        bias2 = pickle.load(fp)
    
    weights = [(weights1, bias1), (weights2, bias2)]

    return weights


def feed_forward(input_, model):
    input_ = np.array(input_).reshape((1,29))
    (weights_1, bias_1), (weights_2, bias_2) = model

    layer1 = np.matmul(input_, weights_1.T)
    layer1 = layer1 + bias_1
    layer1 = np.vectorize(lambda value: max(0, value))(layer1)
    layer2 = np.matmul(layer1, weights_2.T)
    layer2 = layer2 + bias_2

    return layer2.tolist()[0]


def transform_features(features, encoder, scaler):

    non_ohe_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness',
                    'valence', 'tempo', 'duration_ms']
    ohe_columns = ['key', 'mode', 'time_signature']

    
    non_ohe_values = [features[key] for key in non_ohe_columns]
    ohe_values = [features[key] for key in ohe_columns]

    ohe_data = list(non_ohe_values) + list(encoder.transform([ohe_values])[0])

    ohe_minmax = scaler.transform([ohe_data])

    return list(ohe_minmax.tolist()[0])


def knn_search(es, embedding, n=5, existing_songs=set()):
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
        'track_uri': hit['_source']['track_uri'],
        'track_name': hit['_source']['track_name'],
        'track_artist': hit['_source']['track_artist'],
        'id': hit['_id']
    } for hit in resp['hits']['hits'] if hit['_id'] not in existing_songs]

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
                    'track_name': recommended_song['track_name'],
                    'track_artist': recommended_song['track_artist'],
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
    recommendations = [song for song in sorted(inverted_recommendations.values(), key=lambda song: song['score'], reverse=True)][:count]
    recommendations = {song['id']: {
        **song,
        'score': song['score']
    } for song in recommendations}

    return recommendations


def lambda_handler(event, context):

    body = json.loads(event['body'])

    # Initial validation on shape of event
    # Make sure all query fields are present
    for field in QUERY_FIELDS:
        if field not in body:
            return {'statusCode': HTTPStatus.BAD_REQUEST.value}
    
    song_count = body['count']

    es = Elasticsearch(
        DATABASE_URL,
        http_auth=(DATABASE_USER, DATABASE_PASS)
    )

    oheObj, minMaxScalerObj = load_encoders()

    ff_model = retrieve_model()
    song_embeddings = dict()

    # Check which songs already exist in elasticsearch, if not compute new embedding
    for song in body['songs']:
        song_id = song['id']
        try:
            resp = es.get(index=DATABASE_INDEX, id=song_id)
            song_embeddings[song_id] = resp['_source']['embedding']
        except:
            song_embeddings[song_id] = feed_forward(transform_features(song['features'], oheObj, minMaxScalerObj), ff_model)

    playlist_length = len(body['songs'])
    playlist_songs = {song['id'] for song in body['songs']}

    # Do knn search on elasticsearch for each song embedding
    per_song_recommendations = dict()
    for song_id in song_embeddings:
        per_song_recommendations[song_id] = knn_search(es, song_embeddings[song_id], n=playlist_length, existing_songs=playlist_songs)

    # Merge per song recommendations into a list of 'song_count' length that contains score per original input song
    recommendations = calculate_recommendations(song_count, per_song_recommendations)

    edges = []
    for recommended_song in recommendations:
        for source_song_id, source_song in recommendations[recommended_song]['sources'].items():
            edges.append({
                'source': source_song_id,
                'target': recommended_song,
                'weight': source_song['score']
            })
    graph_model = {
        'source_nodes': body['songs'],
        'recommendation_nodes': [{'id': rec['id'], 'score': rec['score'], 'track_name': rec['track_name'], 'track_artist': rec['track_artist']} for rec in recommendations.values()],
        'edges': edges
    }

    return {
        'statusCode': HTTPStatus.OK.value,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(graph_model)
    }