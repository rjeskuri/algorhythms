import torch
import pandas as pd
from annoy import AnnoyIndex

def calculate_similarity_annoy(embedding_path, new_songs_path, top_k, output_path):
    # Load embeddings and new songs from PyTorch files
    embeddings = torch.load(embedding_path)
    new_songs = torch.load(new_songs_path)

    # Convert tensors to numpy arrays
    embeddings = embeddings.numpy()
    new_songs = new_songs.numpy()

    # Get the size of the embeddings
    f = embeddings.shape[1]

    # Build the Annoy index
    t = AnnoyIndex(f, 'angular')  # 'angular' uses the cosine distance
    for i, emb in enumerate(embeddings):
        t.add_item(i, emb)
    t.build(10)  # 10 trees - increase this for more precision

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
#calculate_similarity_annoy('embeddings.pt', 'new_songs.pt', 5, 'models/data/')

