import cv2
from deepface import DeepFace
from insightface.app import FaceAnalysis

# --------------------------
# Initialize SCRFD
# --------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    success, frame = camera.read()

    if not success:
        break

    faces = app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        face_crop = frame[y1:y2, x1:x2]

        cv2.imshow("Face Crop", face_crop)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("v"):

            print("Verifying...")

            try:

                result = DeepFace.verify(
                    img1_path=face_crop,
                    img2_path="known_faces/Gautam_face.jpg",
                    detector_backend="skip",
                    model_name="VGG-Face",
                )

                print(result)

            except Exception as e:

                print("Verification Failed")
                print(e)

    cv2.imshow("SCRFD Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()