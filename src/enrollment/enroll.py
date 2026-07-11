import cv2
from insightface.app import FaceAnalysis

# Initialize SCRFD
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

# Read reference image
image = cv2.imread("reference_images/hemank.png")

faces = app.get(image)

if len(faces) == 0:
    print("No face detected!")
    exit()

# Assume first detected face
face = faces[0]

x1, y1, x2, y2 = face.bbox.astype(int)

# Crop face
face_crop = image[y1:y2, x1:x2]

# Save cropped face
cv2.imwrite("known_faces/hemank_face.jpg", face_crop)

print("Enrollment Successful!")

cv2.imshow("Enrolled Face", face_crop)

cv2.waitKey(0)

cv2.destroyAllWindows()