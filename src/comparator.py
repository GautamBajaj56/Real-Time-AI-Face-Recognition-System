from math import sqrt

def compare_embeddings(embedding1, embedding2):

    distance = 0

    for i in range(len(embedding1)):
        distance += (embedding1[i] - embedding2[i]) ** 2

    return sqrt(distance)