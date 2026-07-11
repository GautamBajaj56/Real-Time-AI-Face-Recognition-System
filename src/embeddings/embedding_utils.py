

from deepface import DeepFace


def generate_embedding(image):
    """
    Generate a face embedding from an image.
    """

    result = DeepFace.represent(
    img_path=image
    )



    
    return result[0]["embedding"]

    