
import React, { useCallback, useState } from "react";

import { Graph } from "react-d3-graph";


const RecommendationGraph = ({ recommendations, setPlaybackSong }) => {

    const [height, setHeight] = useState(null);
    const [width, setWidth] = useState(null);
    const div = useCallback(node => {
        if (node !== null) {
            setHeight(node.getBoundingClientRect().height);
            setWidth(node.getBoundingClientRect().width);
        }
    }, []);

    const data = () => {
        if (recommendations == null || recommendations.length == 0) {
            return { nodes: [], links: [] }
        }

        const recommendedIds = recommendations.edges.map((edge) => edge.source).concat(recommendations.edges.map((edge) => edge.target));

        const sourceNodes = recommendations.source_nodes.map((track) => {
            return (
                {
                    id: track.id,
                    text: `${track.track_name} - ${track.track_artist}`
                }
            )
        }).filter((node) => recommendedIds.includes(node.id));

        const recommendationNodes = recommendations.recommendation_nodes.map((track) => {
            return (
                {
                    id: track.id,
                    text: `${track.track_name} - ${track.track_artist}`,
                    color: "#5E2E3E"
                }
            )
        }).filter((node) => recommendedIds.includes(node.id));

        return {
            nodes: sourceNodes.concat(recommendationNodes),
            links: recommendations.edges
        }
    };

    const myConfig = {
        nodeHighlightBehavior: false,
        width: width,
        height: height,
        gravity: -100000,
        linkStrength: 0,
        node: {
            color: "#DBD3D8",
            size: 1000,
            fontSize: 14,
            highlightStrokeColor: "lightgreen",
            labelPosition: "center",
            labelProperty: "text"
        },
        link: {
            color: "#A7B5B9",
            highlightColor: "black",
        },
    };

    const onClickNode = function(nodeId, node) {
        setPlaybackSong(nodeId);
    };

    const onClickLink = function(source, target) {
        window.alert(`Clicked link between ${source} and ${target}`);
    };

    return (
        <div ref={div} style={{width: '100%', height: '80vh', marginTop: '30px', marginBottom: '-30px', marginRight: '-30px', backgroundColor: '#2E5E4E', borderRadius: '10px'}}>
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