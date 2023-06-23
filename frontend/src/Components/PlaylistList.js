import ListGroup from 'react-bootstrap/ListGroup';

const PlaylistList = ({ playlists, setActivePlaylist }) => {
    const items = () => {
        return playlists.map(playlist => {
            <ListGroup.Item action onClick={setActivePlaylist(playlist.id)}>
                {playlist.name}
            </ListGroup.Item>
        });
    }

    return (
        <ListGroup variant='flush'>
            {items()}
        </ListGroup>
    );
};

export default PlaylistList;