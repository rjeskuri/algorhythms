
import React, { useState, useEffect } from 'react';
import axios from 'axios';


const SpotifyInterface = ({ activePlaylist, setPlaylists }) => {
    const CLIENT_ID = 'b2363da7847344cfa904d18c67075f76';
    const REDIRECT_URI = 'http://localhost:3000';
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
    }, [token]);

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
            setPlaylists(response.items);
            console.log(response);
        }).catch((err) => console.log(err));
    };

    const retrieveActivePlaylist = () => {

    };

    return (
        <div className="App">
            <header className="App-header">
                {!token ?
                    <a href={`${AUTH_ENDPOINT}?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&response_type=${RESPONSE_TYPE}`}>Login
                        to Spotify</a>
                    : <button onClick={logout}>Logout</button>}
            </header>
        </div>
    );
};

export default SpotifyInterface;