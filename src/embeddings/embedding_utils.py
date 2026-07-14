from deepface import DeepFace


def generate_embedding(image):
    """
    Generate ArcFace embedding from an already cropped face.

    Parameters
    ----------
    image : numpy.ndarray
        Face crop obtained from SCRFD.

    Returns
    -------
    list
        Face embedding vector.
    """

    result = DeepFace.represent(
        img_path=image,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False,
    )

    return result[0]["embedding"]