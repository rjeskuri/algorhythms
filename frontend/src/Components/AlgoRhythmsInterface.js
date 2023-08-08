
import React, { useEffect } from 'react';
import axios from 'axios';


const AlgoRhythmsInterface = ({ playlistTracks, setRecommendations }) => {
    const ALGORHYTHMS_ASK_ENDPOINT = 'https://ynegl80fpg.execute-api.us-east-1.amazonaws.com/default/algorhythmsAsk';

    useEffect(() => {
        askRecommendations();
    }, [playlistTracks]);

    const askRecommendations = () => {
        if (playlistTracks == null || playlistTracks.length == 0) {
            return;
        }

        const data = {
            'count': 5,
            'songs': playlistTracks
        }
        const request = {
            method: 'POST',
            //mode: 'no-cors',
            body: JSON.stringify(data),
        };

        (async () => {
            const rawResponse = await fetch(new Request(ALGORHYTHMS_ASK_ENDPOINT, request));
            const content = await rawResponse.json();
            setRecommendations(content)
        })();
    };

    return (
        <div />
    );
};

export default AlgoRhythmsInterface;