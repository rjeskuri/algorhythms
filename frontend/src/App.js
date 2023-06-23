
import React, { useState, useEffect } from 'react';

import PlaylistList from './Components/PlaylistList';
import RecommendationGraph from './Components/RecommendationGraph';
import SpotifyInterface from './Components/SpotifyInterface';

import './App.css';


function App() {
  const [playlists, setPlaylists] = useState([]);
  const [activePlaylist, setActivePlaylist] = useState("");

  return (
    <div className="App">
      <header className="App-header">
        <PlaylistList 
          playlists={playlists}
          setActivePlaylist={setActivePlaylist}
        />
        <SpotifyInterface 
          setPlaylists={setPlaylists}
          activePlaylist={activePlaylist}
        />
        <RecommendationGraph />
      </header>
    </div>
  );
}

export default App;
