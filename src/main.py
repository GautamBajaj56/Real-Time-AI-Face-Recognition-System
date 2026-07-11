import cv2
import time
import psutil
import threading
from insightface.app import FaceAnalysis
from rich.prompt import result
from recognition.recognizer import recognize_face

# ----------------------------------------
# Initialize SCRFD
# ----------------------------------------
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

app.prepare(ctx_id=0, det_size=(640, 640))
# ----------------------------------------
# Open Webcam
# ----------------------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ----------------------------------------
# Recognition State
# ----------------------------------------
recognition_status = None
recognition_distance = None
recognition_time = 0
status_message = ""

# ----------------------------------------
# Background Recognition
# ----------------------------------------
recognition_running = False
recognition_lock = threading.Lock()

# ----------------------------------------
# SCRFD Optimization
# ----------------------------------------
frame_count = -1
last_faces = []
last_verification_time = 0
VERIFICATION_INTERVAL = 1.0
# ----------------------------------------
# Performance Metrics
# ----------------------------------------

last_frame_time = time.perf_counter()
current_fps = 0.0

process = psutil.Process()
# ----------------------------------------
# CPU Metrics
# ----------------------------------------

total_cpu_usage = 0
cpu_samples = 0

# ----------------------------------------
# RAM Metrics
# ----------------------------------------

total_ram_usage = 0
ram_samples = 0

# ----------------------------------------
# Background Verification Function
# ----------------------------------------

def recognize_face_thread(face_image):

    global recognition_running
    global recognition_status
    global recognition_distance
    global recognition_time
    global status_message

    verify_start = time.perf_counter()

    try:

        result = recognize_face(
            face_image,
            "embeddings/embeddings.pkl"
        )

        verify_elapsed = time.perf_counter() - verify_start

        with recognition_lock:

            recognition_status = result["recognized"]
            recognition_distance = result["distance"]
            recognition_time = verify_elapsed

            if recognition_status:
                status_message = result["name"]
            else:
                status_message = "UNKNOWN"

    except Exception as e:

        print("Verification Failed!")
        print(e)

        with recognition_lock:

            recognition_status = None
            recognition_distance = None
            recognition_time = 0
            status_message = "ERROR"

    finally:
        with recognition_lock:
            recognition_running = False


while True:

    success, frame = camera.read()

    if not success:
        break

    # ----------------------------------------
    # Run SCRFD every 8th frame
    # ----------------------------------------
    frame_count += 1

    if frame_count % 8 == 0:
        last_faces = app.get(frame)

    # ----------------------------------------
    # Detect Face & Crop
    # ----------------------------------------
    face_crop = None
    largest_face = None
    largest_area = -1

    for face in last_faces:

        x1, y1, x2, y2 = face.bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area > largest_area:
            largest_area = area
            largest_face = face

    if largest_face is not None:
        x1, y1, x2, y2 = largest_face.bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_crop = frame[y1:y2, x1:x2]

    # ----------------------------------------
    # Automatic Verification (Background Thread)
    # ----------------------------------------
    if time.time() - last_verification_time >= VERIFICATION_INTERVAL:
        last_verification_time = time.time()

        if face_crop is not None:

            # Only start a new verification if one isn't already running
            with recognition_lock:
                should_start = not recognition_running

                if should_start:
                    recognition_running = True

            if should_start:

                # Copy the face because the webcam frame changes every loop
                face_copy = face_crop.copy()

                threading.Thread(
                    target=recognize_face_thread,
                    args=(face_copy,),
                    daemon=True,
                ).start()

        else:

            with recognition_lock:
                recognition_status = None
                recognition_distance = None
                status_message = "NO FACE"

    # ----------------------------------------
    # Draw Recognition Result
    # ----------------------------------------
    with recognition_lock:
        current_distance = recognition_distance
        current_time = recognition_time
        current_message = status_message

    if current_message != "":

        if current_message == "UNKNOWN":
            color = (0, 0, 255)

        elif current_message == "NO FACE":
            color = (0, 255, 255)

        elif current_message == "ERROR":
            color = (0, 165, 255)

        else:
            color = (0, 255, 0)

        cv2.putText(
            frame, current_message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        if current_distance is not None:

            cv2.putText(
                frame,
                f"Distance: {current_distance:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # ----------------------------------------
    # Display Window
    # ----------------------------------------
    current_time_value = time.perf_counter()
    frame_delta = current_time_value - last_frame_time
    last_frame_time = current_time_value

    if frame_delta > 0:
        measured_fps = 1.0 / frame_delta
        current_fps = current_fps * 0.9 + measured_fps * 0.1

    current_cpu = psutil.cpu_percent(interval=None)

    total_cpu_usage += current_cpu
    cpu_samples += 1

    current_ram = process.memory_info().rss / (1024 * 1024)

    total_ram_usage += current_ram
    ram_samples += 1

    average_cpu = total_cpu_usage / cpu_samples
    average_ram = total_ram_usage / ram_samples
    cv2.putText(
        frame,
        f"FPS: {current_fps:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"CPU: {average_cpu:.1f}%",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"RAM: {average_ram:.1f} MB",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Verify: {current_time:.2f}s",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.imshow("Face Verification", frame)

    # ----------------------------------------
    # Quit
    # ----------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
