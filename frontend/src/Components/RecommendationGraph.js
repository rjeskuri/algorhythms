
import React from "react";

import { Graph } from "react-d3-graph";


const RecommendationGraph = ({ tracks, setPlaybackSong }) => {

    const data = () => {
        if (tracks == null || tracks.length == 0) {
            return { nodes: [], links: [] }
        }

        return {
            nodes: tracks.map((track) => {
                return (
                    {
                        id: track.track.name,
                        popularity: track.track.popularity,
                        albumName: track.track.album.name,
                        songId: track.id
                    }
                )
            }),
            links: tracks.flatMap((v, i) =>
                tracks.slice(i + 1).map(w => { return ( {source: v.track.name, target: w.track.name, weight: Math.floor(Math.random() * 30) } ) })
            )
        }
    };

    const myConfig = {
        nodeHighlightBehavior: true,
        width: 1200,
        height: 800,
        node: {
            color: "#DBD3D8",
            size: 1000,
            highlightStrokeColor: "lightgreen",
        },
        link: {
            color: "#A7B5B9",
            highlightColor: "black",
        },
    };

    const onClickNode = function(nodeId, node) {
        window.alert(`Clicked node ${nodeId}`);
        setPlaybackSong(node.songId);
    };

    const onClickLink = function(source, target) {
        window.alert(`Clicked link between ${source} and ${target}`);
    };

    return (
        <div style={{width: '100%', height: '100%', marginTop: '30px', marginBottom: '-30px', marginRight: '-30px', backgroundColor: '#2E5E4E', borderRadius: '10px'}}>
            <Graph
                id="recommendation-graph"
                data={data()}
                config={myConfig}
                onClickNode={onClickNode}
                onClickLink={onClickLink}
            />
        </div>
    )
};

export default RecommendationGraph;