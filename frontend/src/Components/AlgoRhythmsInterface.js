
import React, { useEffect } from 'react';
import axios from 'axios';


const AlgoRhythmsInterface = ({ playlistTracks, setRecommendations }) => {
    const ALGORHYTHMS_ASK_ENDPOINT = '';

    useEffect(() => {
        askRecommendations();
    }, [playlistTracks]);

    const askRecommendations = () => {
        if (playlistTracks == null) {
            return;
        }

        axios.post(ALGORHYTHMS_ASK_ENDPOINT, playlistTracks).then((response) => {
            setRecommendations(response.data.items);
        }).catch((err) => console.log(err));
    };

    return (
        <div />
    );
};

export default AlgoRhythmsInterface;