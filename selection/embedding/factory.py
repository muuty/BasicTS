from selection.embedding.raw import RawEmbedding
from selection.embedding.tsne import TSNEEmbedding
from selection.embedding.umap import UMAPEmbedding




def get_embedding(type: str):
    """Get embedding instance.

    Args:
        type (str): Embedding type.

    Returns:
        BaseEmbedding: Embedding instance.
    """
    if type == 'raw':
        return RawEmbedding()
    if type == 'tsne':
        return TSNEEmbedding()
    if type == 'umap':
        return UMAPEmbedding()
    else:
        raise ValueError('Unknown embedding type!')