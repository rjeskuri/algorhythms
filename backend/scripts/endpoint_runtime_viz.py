
import random
import requests
import time
import json

from matplotlib import pyplot as plt
import numpy as np 


ENDPOINT_URL = 'https://ynegl80fpg.execute-api.us-east-1.amazonaws.com/default/algorhythmsAsk'
SONGS = [
    {
        "features": {
            "danceability": 0.542,
            "energy": 0.88,
            "key": 11,
            "loudness": -4.081,
            "mode": 1,
            "speechiness": 0.0476,
            "acousticness": 0.0341,
            "instrumentalness": 0.0000975,
            "liveness": 0.393,
            "valence": 0.434,
            "tempo": 117.986,
            "duration_ms": 253653,
            "time_signature": 4
        },
        "id": "spotify:track:4xDov3LqBL5j7opAPb0PoD"
    },
    {
        "features": {
            "danceability": 0.557,
            "energy": 0.824,
            "key": 10,
            "loudness": -6.947,
            "mode": 1,
            "speechiness": 0.0776,
            "acousticness": 0.0018,
            "instrumentalness": 0.000542,
            "liveness": 0.0999,
            "valence": 0.611,
            "tempo": 135.446,
            "duration_ms": 245360,
            "time_signature": 4
        },
        "id": "spotify:track:7KEajVD9AUG68onXRSFZTv"
    },
    {
        "features": {
            "danceability": 0.517,
            "energy": 0.918,
            "key": 4,
            "loudness": -3.957,
            "mode": 1,
            "speechiness": 0.059,
            "acousticness": 0.00198,
            "instrumentalness": 0.000677,
            "liveness": 0.123,
            "valence": 0.83,
            "tempo": 151.979,
            "duration_ms": 284547,
            "time_signature": 4
        },
        "id": "spotify:track:5ubUprLRov5UPDCS2UP4cb"
    },
    {
        "features": {
            "danceability": 0.686,
            "energy": 0.93,
            "key": 9,
            "loudness": -3.524,
            "mode": 0,
            "speechiness": 0.0312,
            "acousticness": 0.0401,
            "instrumentalness": 0.000713,
            "liveness": 0.147,
            "valence": 0.782,
            "tempo": 117.035,
            "duration_ms": 213773,
            "time_signature": 4
        },
        "id": "spotify:track:07SgR5jvYFyVDHa6uv8Fh0"
    },
    {
        "features": {
            "danceability": 0.441,
            "energy": 0.87,
            "key": 2,
            "loudness": -4.845,
            "mode": 1,
            "speechiness": 0.057,
            "acousticness": 0.0533,
            "instrumentalness": 0.0000037,
            "liveness": 0.0936,
            "valence": 0.274,
            "tempo": 134.469,
            "duration_ms": 225160,
            "time_signature": 4
        },
        "id": "spotify:track:1MrqlDkxsrMsxHni2Yb9Tx"
    },
    {
        "features": {
            "danceability": 0.619,
            "energy": 0.92,
            "key": 5,
            "loudness": -4.374,
            "mode": 1,
            "speechiness": 0.0305,
            "acousticness": 0.0142,
            "instrumentalness": 0.000206,
            "liveness": 0.253,
            "valence": 0.424,
            "tempo": 99.985,
            "duration_ms": 218893,
            "time_signature": 4
        },
        "id": "spotify:track:4T3fEhOYsYTwtxzA7TQBkf"
    },
    {
        "features": {
            "danceability": 0.587,
            "energy": 0.844,
            "key": 8,
            "loudness": -7.061,
            "mode": 1,
            "speechiness": 0.0984,
            "acousticness": 0.196,
            "instrumentalness": 0,
            "liveness": 0.178,
            "valence": 0.652,
            "tempo": 94.014,
            "duration_ms": 185360,
            "time_signature": 4
        },
        "id": "spotify:track:36GPL4a7IHf0C6DdBpoifS"
    },
    {
        "features": {
            "danceability": 0.388,
            "energy": 0.813,
            "key": 11,
            "loudness": -5.364,
            "mode": 1,
            "speechiness": 0.039,
            "acousticness": 0.0672,
            "instrumentalness": 0,
            "liveness": 0.378,
            "valence": 0.501,
            "tempo": 173.913,
            "duration_ms": 184133,
            "time_signature": 4
        },
        "id": "spotify:track:5wXsFe8dNMHOau3WyfKuVZ"
    },
    {
        "features": {
            "danceability": 0.594,
            "energy": 0.87,
            "key": 0,
            "loudness": -5.308,
            "mode": 1,
            "speechiness": 0.0259,
            "acousticness": 0.0229,
            "instrumentalness": 0.00697,
            "liveness": 0.312,
            "valence": 0.579,
            "tempo": 105.994,
            "duration_ms": 278493,
            "time_signature": 4
        },
        "id": "spotify:track:6M0VByRzxb5ycM9QcMx9k5"
    },
    {
        "features": {
            "danceability": 0.38,
            "energy": 0.52,
            "key": 11,
            "loudness": -8.494,
            "mode": 0,
            "speechiness": 0.0359,
            "acousticness": 0.0187,
            "instrumentalness": 0.36,
            "liveness": 0.101,
            "valence": 0.0801,
            "tempo": 128.044,
            "duration_ms": 411373,
            "time_signature": 4
        },
        "id": "spotify:track:0HfMp4RGDgjvrWX3j0VZnJ"
    }
]


def query_backend(query):

    start_time = time.time()

    resp = requests.post(url=ENDPOINT_URL, data=json.dumps(query))

    query_time = time.time() - start_time
    return query_time


def main():
    
    # Retrieve data for varying number of songs in base dataset
    playlist_length_data = {}
    for i in range(1, len(SONGS)):
        playlist_length_data[i] = []
        for _ in range(25):
            songs = random.sample(SONGS, i)
            query = {
                'count': 5,
                'songs': songs
            }
            playlist_length_data[i].append(query_backend(query))
            print(f'{i} - {_}')
    

    # Retrieve data for varying number of requested recommendations
    recommendation_length_data = {}
    for i in range(1, 11):
        recommendation_length_data[i] = []
        for _ in range(25):
            query = {
                'count': i,
                'songs': SONGS
            }
            recommendation_length_data[i].append(query_backend(query))
            print(f'{i} - {_}')
    
    playlist_data = [np.mean(data) for data in playlist_length_data.values()]
    playlist_errors = [np.std(data) for data in playlist_length_data.values()]
    playlist_x_labels = [str(label) for label in playlist_length_data.keys()]
    playlist_x_pos = np.arange(len(playlist_x_labels))
    recommendation_data = [np.mean(data) for data in recommendation_length_data.values()]
    recommendation_errors = [np.std(data) for data in recommendation_length_data.values()]
    recommendation_x_labels = [str(label) for label in recommendation_length_data.keys()]
    recommendation_x_pos = np.arange(len(recommendation_x_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Backend runtime performance')

    ax1.bar(playlist_x_pos, playlist_data, yerr=playlist_errors, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.set_ylabel('POST Request Runtime (s)')
    ax1.set_xticks(playlist_x_pos)
    ax1.set_xticklabels(playlist_x_labels)
    ax1.set_title('Length of query playlist')
    ax1.yaxis.grid(True)

    ax2.bar(recommendation_x_pos, recommendation_data, yerr=recommendation_data, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax2.set_ylabel('POST Request Runtime (s)')
    ax2.set_xticks(recommendation_x_pos)
    ax2.set_xticklabels(recommendation_x_labels)
    ax2.set_title('Count of requested recommendations')
    ax2.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('runtime_performance.png')
    plt.show()


if __name__ == '__main__':
    main()