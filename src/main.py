import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Store latest verification result
verification_status = None
verification_distance = None

while True:

    success, frame = camera.read()

    if not success:
        break

    key = cv2.waitKey(1)

    if key == ord("v"):

        print("Verifying...")

        result = DeepFace.verify(
            img1_path=frame,
            img2_path="known_faces/Gautam.jpg",
            detector_backend="opencv"
        )

        print("Verification Finished!")

        verification_status = result["verified"]
        verification_distance = result["distance"]

    # Draw latest verification result
    if verification_status is not None:

        if verification_status:

            cv2.putText(
                frame,
                "VERIFIED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        else:

            cv2.putText(
                frame,
                "UNKNOWN",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Distance: {verification_distance:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Face Verification", frame)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()