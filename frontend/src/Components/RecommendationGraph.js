
import React, { useState, useEffect } from "react";

import { Graph } from "react-d3-graph";


const RecommendationGraph = ({}) => {

    // graph payload (with minimalist structure)
    const data = {
        nodes: [
            { id: "Harry" },
            { id: "Sally" },
            { id: "Alice" }
        ],
        links: [
            { source: "Harry", target: "Sally" },
            { source: "Harry", target: "Alice" },
        ],
    };

    // the graph configuration, just override the ones you need
    const myConfig = {
        nodeHighlightBehavior: true,
        node: {
            color: "black",
            size: 1000,
            highlightStrokeColor: "lightgreen",
        },
        link: {
            color: "darkgrey",
            highlightColor: "black",
        },
    };

    const onClickNode = function(nodeId) {
        window.alert(`Clicked node ${nodeId}`);
    };

    const onClickLink = function(source, target) {
        window.alert(`Clicked link between ${source} and ${target}`);
    };

    return (
        <div style={{width: '1500px'}}>
            <Graph
                id="recommendation-graph" // id is mandatory
                data={data}
                config={myConfig}
                onClickNode={onClickNode}
                onClickLink={onClickLink}
            />
        </div>
    )
};

export default RecommendationGraph;