from unittest import result

from deepface import DeepFace


def generate_embedding(image):
    """
    Generate a face embedding from an image.
    """

    result = DeepFace.represent(
    img_path=image
    )



    print(len(result))
    print(result[0].keys())
    return result[0]["embedding"]

    