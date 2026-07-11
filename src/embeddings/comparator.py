import numpy as np


def cosine_distance(embedding1, embedding2):
    """
    Calculate cosine distance between two face embeddings.

    Lower distance = More similar faces.
    """

    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    cosine_similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

    return 1 - cosine_similarity


def find_best_match(live_embedding, stored_embeddings):
    """
    Compare a live embedding against all enrolled embeddings.

    Parameters
    ----------
    live_embedding : list
        Embedding generated from the live camera face.

    stored_embeddings : dict
        {
            "Gautam": [...],
            "Hemank": [...],
            ...
        }

    Returns
    -------
    tuple
        (
            best_name,
            best_distance
        )
    """

    best_name = None
    best_distance = float("inf")

    for person_name, embedding in stored_embeddings.items():

        distance = cosine_distance(
            live_embedding,
            embedding,
        )

        if distance < best_distance:
            best_distance = distance
            best_name = person_name

    return best_name, best_distance