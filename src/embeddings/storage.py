import os
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

    # Return cache if already loaded
    if _cached_embeddings is not None and not force_reload:
        return _cached_embeddings

    # Database doesn't exist yet
    if not os.path.exists(file_path):
        _cached_embeddings = {}
        return _cached_embeddings

    # Database exists but is empty
    if os.path.getsize(file_path) == 0:
        _cached_embeddings = {}
        return _cached_embeddings

    # Load database
    try:
        with open(file_path, "rb") as file:
            _cached_embeddings = pickle.load(file)

    except (EOFError, pickle.UnpicklingError):
        _cached_embeddings = {}

    return _cached_embeddings