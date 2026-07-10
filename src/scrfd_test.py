import cv2
import time
from insightface.app import FaceAnalysis

# Initialize SCRFD
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# FPS calculation
prev_time = time.time()

# Optimization variables
frame_count = 0
last_faces = []

while True:
    success, frame = camera.read()
    if not success:
        break

    # Increment frame counter
    frame_count += 1

    # Run SCRFD only every 5th frame
    if frame_count % 5 == 0:
        last_faces = app.get(frame)

    # Draw last detected faces
    for face in last_faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop only the detected face and show if valid
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size != 0:
            cv2.imshow("Face Crop", face_crop)

    # FPS calculation
    current_time = time.time()
    delta = current_time - prev_time
    fps = 1 / delta if delta > 0 else 0.0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(last_faces)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("SCRFD Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()