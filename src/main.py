import cv2
from embedding_utils import generate_embedding

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, frame = camera.read()

    if not success:
        break

    cv2.imshow("Face Verification", frame)

    key = cv2.waitKey(1)

    if key == ord("e"):

        print("Generating embedding...")

        embedding = generate_embedding(frame)

        print("Embedding Length:", len(embedding))
        print("First 5 Values:", embedding[:5])

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()




