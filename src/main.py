import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    success, frame = camera.read()

    if not success:
        break

    cv2.imshow("Face Verification", frame)

    key = cv2.waitKey(1)

    if key == ord("v"):

        print("Verifying...")

        result = DeepFace.verify(
            img1_path=frame,
            img2_path="known_faces/Gautam.jpg",
            detector_backend="opencv"
        )

        print(result)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()