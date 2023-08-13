import torch
from annoy import AnnoyIndex

def build_and_save_index(embedding_path, output_path, num_trees=10):
    # Load embeddings from PyTorch file
    embeddings = torch.load(embedding_path)

    # Convert tensor to numpy array
    embeddings = embeddings.numpy()

    # Get the size of the embeddings
    f = embeddings.shape[1]

    # Build the Annoy index
    t = AnnoyIndex(f, 'angular')  # 'angular' uses the cosine distance
    for i, emb in enumerate(embeddings):
        t.add_item(i, emb)

    t.build(num_trees)  # num_trees trees - increase this for more precision

    # Save the index
    t.save(output_path + 'annoy_index.ann')

    print("Annoy index has been built and saved!")

# Example of how to call the function
build_and_save_index('embeddings.pt', 'models/data/')
