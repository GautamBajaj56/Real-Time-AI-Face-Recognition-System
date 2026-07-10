import cv2
from deepface import DeepFace
from insightface.app import FaceAnalysis

# -----------------------------
# Initialize SCRFD
# -----------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------
# Open Webcam
# -----------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -----------------------------
# Store latest verification result
# -----------------------------
verification_status = None
verification_distance = None
status_message = ""

# -----------------------------
# SCRFD Optimization Variables
# -----------------------------
frame_count = -1
last_faces = []

while True:

    success, frame = camera.read()

    if not success:
        break

    # ---------------------------------------
    # Run SCRFD every 5th frame
    # ---------------------------------------
    frame_count += 1

    if frame_count % 5 == 0:
        last_faces = app.get(frame)

    # ---------------------------------------
    # Detect & Draw Face
    # ---------------------------------------
    face_crop = None

    for face in last_faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        face_crop = frame[y1:y2, x1:x2]

    # ---------------------------------------
    # Keyboard Input
    # ---------------------------------------
    key = cv2.waitKey(1) & 0xFF

    # ---------------------------------------
    # Verification
    # ---------------------------------------
    if key == ord("v") and face_crop is not None:

        print("Verifying...")

        try:

            result = DeepFace.verify(
                img1_path=face_crop,
                img2_path="known_faces/Gautam_face.jpg",
                detector_backend="skip",
                model_name="VGG-Face"
            )

            print("Verification Finished!")
            print(result)

            verification_status = result["verified"]
            verification_distance = result["distance"]

            if verification_status:
                status_message = "VERIFIED"
            else:
                status_message = "UNKNOWN"

        except Exception as e:

            print("\nVerification Failed!")
            print(e)

            verification_status = None
            verification_distance = None
            status_message = "ERROR"

    elif key == ord("v") and face_crop is None:

        status_message = "NO FACE DETECTED"
        verification_status = None
        verification_distance = None

    # ---------------------------------------
    # Draw Verification Result
    # ---------------------------------------
    if status_message != "":

        if status_message == "VERIFIED":
            color = (0, 255, 0)

        elif status_message == "UNKNOWN":
            color = (0, 0, 255)

        else:
            color = (0, 255, 255)

        cv2.putText(
            frame,
            status_message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        if verification_distance is not None:

            cv2.putText(
                frame,
                f"Distance: {verification_distance:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # ---------------------------------------
    # Display Webcam
    # ---------------------------------------
    cv2.imshow("Face Verification", frame)

    # ---------------------------------------
    # Quit
    # ---------------------------------------
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()