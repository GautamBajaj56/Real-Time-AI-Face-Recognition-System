from storage import load_embeddings
from embedding_utils import generate_embedding
from comparator import compare_embeddings

THRESHOLD = 0.6

def verify_person(file_path, person_name):

    current_embedding = generate_embedding(file_path)

    stored_embedding = load_embeddings(
        "embeddings/embeddings.pkl"
    )[person_name]

    distance = compare_embeddings(
        current_embedding,
        stored_embedding
    )

    if distance <= THRESHOLD:
        return True, distance

    return False, distance