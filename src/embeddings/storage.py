import pickle

_cached_embeddings = None


def save_embeddings(data, file_path):
    """
    Save embeddings dictionary to disk.
    """

    global _cached_embeddings

    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    # Update in-memory cache
    _cached_embeddings = data


def load_embeddings(file_path, force_reload=False):
    """
    Load embeddings from disk.

    Uses an in-memory cache so the file is only read once unless
    force_reload=True.
    """

    global _cached_embeddings

    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    with open(file_path, "rb") as file:
        _cached_embeddings = pickle.load(file)

    return _cached_embeddings