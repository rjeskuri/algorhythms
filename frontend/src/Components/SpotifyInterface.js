
import React, { useState, useEffect } from 'react';
import axios from 'axios';

import Button from 'react-bootstrap/Button';


const SpotifyInterface = ({ activePlaylist, setPlaylists, setTracks, setSpotifyToken }) => {
    const CLIENT_ID = 'b2363da7847344cfa904d18c67075f76';
    const REDIRECT_URI = 'http://algorhythms-frontend.s3-website-us-east-1.amazonaws.com/';
    const AUTH_ENDPOINT = 'https://accounts.spotify.com/authorize';
    const RESPONSE_TYPE = 'token';

    const [token, setToken] = useState("");

    useEffect(() => {
        const hash = window.location.hash;
        let token = window.localStorage.getItem("token");

        if (!token && hash) {
            token = hash.substring(1).split("&").find(elem => elem.startsWith("access_token")).split("=")[1];

            window.location.hash = "";
            window.localStorage.setItem("token", token);
        }

        setToken(token);
    }, []);

    useEffect(() => {
        searchPlaylists();
        setSpotifyToken(token);
    }, [token]);

    useEffect(() => {
        retrieveActivePlaylist();
    }, [activePlaylist]);

    const logout = () => {
        setToken("");
        window.localStorage.removeItem("token");
    };

    const searchPlaylists = () => {
        axios.get("https://api.spotify.com/v1/me/playlists", {
            headers: {
                Authorization: `Bearer ${token}`
            },
            params: {
                
            }
        }).then((response) => {
            var playlists = response.data.items;
            playlists.unshift({
                id: 'liked',
                name: 'Liked Songs'
            });

            setPlaylists(playlists);
        }).catch((err) => console.log(err));
    };

    const retrieveActivePlaylist = () => {
        let queryEndpoint = '';
        if (activePlaylist === '') {
            return;
        } else if (activePlaylist === 'liked') {
            queryEndpoint = `https://api.spotify.com/v1/me/tracks`;
        } else {
            queryEndpoint = `https://api.spotify.com/v1/playlists/${activePlaylist}/tracks`;
        }

        axios.get(queryEndpoint, {
            headers: {
                Authorization: `Bearer ${token}`
            }
        }).then((response) => {
            retrieveAudioFeatures(response.data.items);
        }).catch((err) => console.log(err));
    };

    const retrieveAudioFeatures = (tracks) => {
        const songIds = tracks.map(track => track.track.id).join(',');

        axios.get('https://api.spotify.com/v1/audio-features', {
            params: {
                ids: songIds
            },
            headers: {
                Authorization: `Bearer ${token}`
            }
        }).then((response) => {
            const playlist = tracks.map(function (e, i) {
                return {features: response.data.audio_features[i], id: e.track.uri, track_name: e.track.name, track_artist: e.track.artists[0].name}
            });

            setTracks(playlist)
        })
    };

    return (
        <div style={{marginTop: '30px'}}>
            {!token ?
                <a href={`${AUTH_ENDPOINT}?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&response_type=${RESPONSE_TYPE}&scope=user-library-read,streaming,user-read-email,user-read-private,user-read-playback-state,user-modify-playback-state`}>Login
                    to Spotify</a>
                : <Button variant='secondary' onClick={logout}>Logout</Button>}
        </div>
    );
};

export default SpotifyInterface;