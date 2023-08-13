import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import configparser

# Spotify API client

config_options = configparser.ConfigParser()
conf_dir = os.environ.get('SPARK_CONF_DIR') or 'conf'  # Options to support Spark CLuster and local modes
config_options.read('{}/env-secrets.conf'.format(conf_dir))  # Load entries defined in 'spark-start' shell script


client_id = dict(config_options.items("ENV_SECRETS")).get('spotify_client_id')
client_secret = dict(config_options.items("ENV_SECRETS")).get('spotify_client_secret')
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def getinfotracks(tracks_uri):
    #  tracks_uri needs to be max 50 items in the list
    spotify_track_id_list = [track_uri.split(':')[-1] for track_uri in tracks_uri]  # Extract Spotify trackID from URI
    track_info = sp.tracks(spotify_track_id_list)  # Search for the 'tracks' using a list of Spotify track IDs
    return track_info


def getinfoaudiofeatures(tracks_uri):
    #  tracks_uri needs to be max 50 items in the list
    spotify_track_id_list = [track_uri.split(':')[-1] for track_uri in tracks_uri]  # Extract Spotify trackID from URI
    track_info = sp.audio_features(spotify_track_id_list)  # Search for the 'audio_features' using a list of Spotify track IDs
    return track_info
