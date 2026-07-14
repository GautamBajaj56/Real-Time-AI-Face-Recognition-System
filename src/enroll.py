import cv2
from insightface.app import FaceAnalysis

from embeddings.embedding_utils import generate_embedding
from embeddings.storage import load_embeddings, save_embeddings

# ---------------------------------------------------
# User Input
# ---------------------------------------------------

PERSON_NAME = input("Enter person's name: ").strip()

if not PERSON_NAME:
    raise ValueError("Person name cannot be empty.")

REFERENCE_IMAGE_PATH = input(
    "Enter reference image path: "
).strip()

if not REFERENCE_IMAGE_PATH:
    raise ValueError("Reference image path cannot be empty.")

CROPPED_FACE_PATH = f"known_faces/{PERSON_NAME}_face.jpg"

EMBEDDINGS_PATH = "embeddings/embeddings.pkl"

# ---------------------------------------------------
# Initialize SCRFD
# ---------------------------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640),
)

# ---------------------------------------------------
# Read Image
# ---------------------------------------------------

image = cv2.imread(REFERENCE_IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {REFERENCE_IMAGE_PATH}"
    )

# ---------------------------------------------------
# Detect Face
# ---------------------------------------------------

faces = app.get(image)

if len(faces) == 0:
    raise Exception("No face detected in reference image.")

# ---------------------------------------------------
# Use Largest Face
# ---------------------------------------------------

largest_face = max(
    faces,
    key=lambda face: (
        face.bbox[2] - face.bbox[0]
    )
    * (
        face.bbox[3] - face.bbox[1]
    ),
)

x1, y1, x2, y2 = largest_face.bbox.astype(int)

face_crop = image[y1:y2, x1:x2]

# ---------------------------------------------------
# Save Cropped Face
# ---------------------------------------------------

cv2.imwrite(
    CROPPED_FACE_PATH,
    face_crop,
)

print(f"Cropped face saved to {CROPPED_FACE_PATH}")

# ---------------------------------------------------
# Generate Embedding
# ---------------------------------------------------

embedding = generate_embedding(face_crop)

print("Embedding generated successfully.")

# ---------------------------------------------------
# Load Existing Database
# ---------------------------------------------------

database = load_embeddings(EMBEDDINGS_PATH)

# ---------------------------------------------------
# Add / Update Person
# ---------------------------------------------------

database[PERSON_NAME] = embedding

# ---------------------------------------------------
# Save Database
# ---------------------------------------------------

save_embeddings(
    database,
    EMBEDDINGS_PATH,
)

print(f"{PERSON_NAME} enrolled successfully.")

print(f"Total enrolled identities: {len(database)}")

# ---------------------------------------------------
# Preview
# ---------------------------------------------------

cv2.imshow(
    "Enrolled Face",
    face_crop,
)

cv2.waitKey(0)

cv2.destroyAllWindows()