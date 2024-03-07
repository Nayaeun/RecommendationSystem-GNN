
# %%

import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_sparse import SparseTensor

#%%
# Assuming the 'data' directory is in the same directory as your script
data_dir = './data'

# URLs for the datasets you want to use
movie_lens_url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

# Define where to download and extract datasets
download_path = f'{data_dir}/ml-latest-small.zip'
extract_dir = f'{data_dir}/ml-latest-small'

# Download and extract dataset
extract_zip(download_url(movie_lens_url, data_dir), data_dir)

# Define paths to data
movie_path = f'{extract_dir}/movies.csv'
rating_path = f'{extract_dir}/ratings.csv'


# Check if the data directory exists, and create it if it doesn't
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Check if the dataset zip file exists before attempting to download
if not os.path.isfile(download_path):
    extract_zip(download_url(movie_lens_url, data_dir), data_dir)



# %%
class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()
# %%
class GenresEncoder:
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x
    
#%%
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.tensor(df.values, dtype=self.dtype).view(-1, 1)


#%%
# Load the node features
movie_x, movie_mapping = load_node_csv(
    movie_path, index_col='movieId', encoders={
        'title': SequenceEncoder(),  # Replace with actual encoder implementations
        'genres': GenresEncoder()    # Replace with actual encoder implementations
    }
)

# Load user nodes
_, user_mapping = load_node_csv(rating_path, index_col='userId')

# Load the edge data
edge_index, edge_label = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.float)}  # Replace with actual encoder implementations
)

# Now, define the function to convert edge index to adjacency matrix
def edge_index_to_adjacency(edge_index, num_nodes):
    # Create a SparseTensor and then convert it to a dense matrix
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    adj = adj.to_dense()
    return adj

# Finally, call the function after you've loaded your graph data
data = HeteroData()
data['movie'].x = movie_x
data['user'].num_nodes = len(user_mapping)
data['user', 'rates', 'movie'].edge_index = edge_index
data['user', 'rates', 'movie'].edge_attr = edge_label

# Calculate the adjacency matrix after loading the graph data
num_users = data['user'].num_nodes
num_movies = data['movie'].x.size(0)
adj = edge_index_to_adjacency(data['user', 'rates', 'movie'].edge_index, num_users + num_movies)

# %%
