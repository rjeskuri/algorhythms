import torch
import pandas as pd
from annoy import AnnoyIndex

#need to generate new song embeddings

def load_index_and_query(new_songs_path, index_path, top_k, output_path):
    # Load new songs from PyTorch file
    new_songs = torch.load(new_songs_path)

    # Convert tensor to numpy array
    new_songs = new_songs.numpy()

    # Get the size of the new songs embeddings
    f = new_songs.shape[1]

    # Load the Annoy index
    t = AnnoyIndex(f, 'angular')  # 'angular' uses the cosine distance
    t.load(index_path)  # super fast, will just mmap the file

    # Initialize lists to store indices and similarities
    top_k_indices = []
    top_k_similarities = []

    # For each song
    for new_song in new_songs:
        indices, similarities = t.get_nns_by_vector(new_song, top_k, include_distances=True)

        top_k_indices.append(indices)
        top_k_similarities.append(similarities)

    # Convert to DataFrames
    df_similarities = pd.DataFrame(top_k_similarities)
    df_indices = pd.DataFrame(top_k_indices)

    # Concatenate DataFrames along the horizontal axis
    df_merged = pd.concat([df_indices, df_similarities], axis=1)

    # Save DataFrames
    df_similarities.to_csv(output_path + 'similarities.csv', index=False)
    df_indices.to_csv(output_path + 'indices.csv', index=False)
    df_merged.to_csv(output_path + 'merged.csv', index=False)

    print("Top K similarities, indices and merged data have been saved!")

# Example of how to call the function
load_index_and_query('new_songs.pt', 'models/data/annoy_index.ann', 5, 'models/data/')

#logic check to make sure not returning songs already in the playlist. 