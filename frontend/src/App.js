
import React, { useState, useEffect } from 'react';

import PlaylistList from './Components/PlaylistList';
import RecommendationGraph from './Components/RecommendationGraph';
import SpotifyInterface from './Components/SpotifyInterface';

import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

import './App.css';
import 'bootstrap/dist/css/bootstrap.css';


function App() {
  const [playlists, setPlaylists] = useState([]);
  const [activePlaylist, setActivePlaylist] = useState("");
  const [tracks, setTracks] = useState([]);

  return (
    <div className="App">
      <div className='App-header'>AlgoRhythms</div>
      <Container fluid style={{display: 'block', width: '90%'}}>
        <Row>
          <Col>
            <Row>
              <SpotifyInterface 
                setPlaylists={setPlaylists}
                activePlaylist={activePlaylist}
                setTracks={setTracks}
              />
            </Row>
            <Row>
              <PlaylistList 
                playlists={playlists}
                setActivePlaylist={setActivePlaylist}
              />
            </Row>
          </Col>
          <Col xs={8}>
            <RecommendationGraph 
              tracks={tracks}
            />
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;
