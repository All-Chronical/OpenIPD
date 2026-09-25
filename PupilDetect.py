import os
import urllib.request
import cv2
import mediapipe as mp

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Pupil / iris center landmark indices in MediaPipe Face Landmarker
LEFT_PUPIL_INDEX = 468
RIGHT_PUPIL_INDEX = 473


def ensure_model_exists():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Downloading {MODEL_FILENAME}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class PupilDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = ensure_model_exists()

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect_pupils(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]
        left_p = landmarks[LEFT_PUPIL_INDEX]
        right_p = landmarks[RIGHT_PUPIL_INDEX]

        left_pupil = (int(left_p.x * w), int(left_p.y * h))
        right_pupil = (int(right_p.x * w), int(right_p.y * h))

        return left_pupil, right_pupil

    def close(self):
        self.landmarker.close()


_detector = None


def detect_pupils(frame):
    global _detector
    if _detector is None:
        _detector = PupilDetector()
    return _detector.detect_pupils(frame)

