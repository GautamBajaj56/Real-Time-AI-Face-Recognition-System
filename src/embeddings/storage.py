import pickle


def save_embeddings(data, file_path):
    """
    Save embeddings dictionary to disk.
    """

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_embeddings(file_path):
    """
    Load embeddings dictionary from disk.
    """

    with open(file_path, "rb") as file:
        return pickle.load(file)