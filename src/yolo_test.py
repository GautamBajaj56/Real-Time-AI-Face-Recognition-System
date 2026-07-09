import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("face_model.pt")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Loop through every detected object
    for box in results[0].boxes:

        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]

        # Convert tensor values to integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cropped = frame[y1:y2, x1:x2]

        cv2.imshow("Cropped", cropped)
        cv2.imshow("YOLO Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()