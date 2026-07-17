import cv2
import numpy as np
import onnxruntime as ort


class AntiSpoofModel:
    """
    MiniFASNet ONNX wrapper for face anti-spoofing.

    Expected input:
        Cropped face image (RGB)

    Output:
        (is_live, confidence)
    """

    def __init__(self, model_path: str):
        self.model_size = 128

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def crop_face(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        expansion_factor: float = 1.5,
    ) -> np.ndarray:
        """
        Extract the square expanded crop used by the original ONNX demo.
        """

        original_height, original_width = image.shape[:2]
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            raise ValueError("Invalid bbox dimensions")

        max_dim = max(width, height)
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        crop_x = int(center_x - max_dim * expansion_factor / 2)
        crop_y = int(center_y - max_dim * expansion_factor / 2)
        crop_size = int(max_dim * expansion_factor)

        src_x1 = max(0, crop_x)
        src_y1 = max(0, crop_y)
        src_x2 = min(original_width, crop_x + crop_size)
        src_y2 = min(original_height, crop_y + crop_size)

        top_pad = int(max(0, -crop_y))
        left_pad = int(max(0, -crop_x))
        bottom_pad = int(max(0, (crop_y + crop_size) - original_height))
        right_pad = int(max(0, (crop_x + crop_size) - original_width))

        if src_x2 > src_x1 and src_y2 > src_y1:
            crop = image[src_y1:src_y2, src_x1:src_x2, :]
        else:
            crop = np.zeros((0, 0, 3), dtype=image.dtype)

        crop = cv2.copyMakeBorder(
            crop,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_REFLECT_101,
        )

        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            raise ValueError(
                f"Crop size mismatch: expected {crop_size}x{crop_size}, "
                f"got {crop.shape[0]}x{crop.shape[1]}"
            )

        return crop

    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Resize using the same letterboxing strategy as the original repository.
        """

        old_h, old_w = face.shape[:2]

        ratio = self.model_size / max(old_h, old_w)

        new_h = int(old_h * ratio)
        new_w = int(old_w * ratio)

        interpolation = (
            cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        )

        face = cv2.resize(
            face,
            (new_w, new_h),
            interpolation=interpolation,
        )

        delta_w = self.model_size - new_w
        delta_h = self.model_size - new_h

        top = delta_h // 2
        bottom = delta_h - top

        left = delta_w // 2
        right = delta_w - left

        face = cv2.copyMakeBorder(
            face,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )

        face = face.transpose(2, 0, 1).astype(np.float32)
        face /= 255.0

        face = np.expand_dims(face, axis=0)

        return face

    def predict(self, face: np.ndarray) -> np.ndarray:
        """
        Returns raw logits:
        [real_logit, spoof_logit]
        """

        input_tensor = self.preprocess(face)

        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]

        return output[0]

    def check(self, face: np.ndarray, threshold: float = 0.5):
        """
        Returns:
            is_live (bool)
            confidence (float)
        """

        logits = self.predict(face)

        real_logit = float(logits[0])
        spoof_logit = float(logits[1])

        logit_difference = real_logit - spoof_logit

        p = max(1e-6, min(1 - 1e-6, threshold))
        logit_threshold = np.log(p / (1 - p))

        is_live = logit_difference >= logit_threshold

        confidence = abs(logit_difference)

        print(
    f"[AntiSpoof] "
    f"Real={real_logit:.3f} | "
    f"Spoof={spoof_logit:.3f} | "
    f"Diff={logit_difference:.3f} | "
    f"Live={is_live}"
)

        return is_live, confidence
    
#     def check(self, face: np.ndarray, threshold: float = 0.5):

#         logits = self.predict(face)

#         print(
#         f"Real={real_logit:.3f}, "
#         f"Spoof={spoof_logit:.3f}, "
#         f"Diff={real_logit - spoof_logit:.3f}, "
#         f"Live={is_live}"
# )

#         return False, 0
    
    
