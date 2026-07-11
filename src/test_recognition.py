from recognition.recognizer import recognize_face
import cv2

# ----------------------------------------
# Test Image
# ----------------------------------------

image = cv2.imread("test_images/gautam_test.png")

# ----------------------------------------
# Recognition
# ----------------------------------------

result = recognize_face(
    face_image=image,
    embeddings_path="embeddings/embeddings.pkl",
)

# ----------------------------------------
# Print Result
# ----------------------------------------

print("\nRecognition Result")
print("----------------------------")
print(f"Recognized : {result['recognized']}")
print(f"Name       : {result['name']}")
print(f"Distance   : {result['distance']}")