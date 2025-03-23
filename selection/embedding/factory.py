from selection.embedding.raw import RawEmbedding
from selection.embedding.tsne import TSNEEmbedding
from selection.embedding.pca import PCAEmbedding
from selection.embedding.umap import UMAPEmbedding




def get_embedding(type: str):
    """Get embedding instance.

    Args:
        type (str): Embedding type.

    Returns:
        BaseEmbedding: Embedding instance.
    """
    if not type:
        return None

    if type == 'raw':
        return RawEmbedding()
    if type == 'tsne':
        return TSNEEmbedding()
    if type == 'umap':
        return UMAPEmbedding()
    if type.startswith('pca'):
        n_components = int(type.split('_')[1])
        return PCAEmbedding(n_components=n_components)
    else:
        raise ValueError('Unknown embedding type!')