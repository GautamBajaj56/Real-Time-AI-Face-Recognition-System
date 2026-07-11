from embeddings.embedding_utils import generate_embedding
from embeddings.storage import load_embeddings
from embeddings.comparator import find_best_match


# ArcFace threshold
# Can later be moved to a config file
ARCFACE_THRESHOLD = 0.68


def recognize_face(face_image, embeddings_path):
    """
    Recognize a face from a live image.

    Parameters
    ----------
    face_image
        Cropped face image (NumPy array).

    embeddings_path
        Path to embeddings.pkl

    Returns
    -------
    dict

    Example
    -------
    {
        "recognized": True,
        "name": "Gautam",
        "distance": 0.42
    }

    OR

    {
        "recognized": False,
        "name": "Unknown",
        "distance": 0.81
    }
    """

    # ----------------------------------------
    # Generate embedding for current face
    # ----------------------------------------
    live_embedding = generate_embedding(face_image)

    # ----------------------------------------
    # Load enrolled embeddings
    # ----------------------------------------
    stored_embeddings = load_embeddings(embeddings_path)

    # ----------------------------------------
    # Find nearest enrolled person
    # ----------------------------------------
    best_name, best_distance = find_best_match(
        live_embedding,
        stored_embeddings,
    )

    # ----------------------------------------
    # Threshold Decision
    # ----------------------------------------
    if best_distance <= ARCFACE_THRESHOLD:

        return {
            "recognized": True,
            "name": best_name,
            "distance": round(best_distance, 4),
        }

    return {
        "recognized": False,
        "name": "Unknown",
        "distance": round(best_distance, 4),
    }