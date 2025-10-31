import argparse
import threading
from abc import ABC, abstractmethod
from queue import Queue

import cv2

from aqara_video.core.types import ImageCV
from aqara_video.ml.detector import Detector
from aqara_video.ml.utils import Prediction, draw_boxes


class VideoAbstract(ABC):
    @abstractmethod
    def capture(self) -> ImageCV:
        pass


class VideoOpenCV(VideoAbstract):
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = self.open_device(camera_id)

    def open_device(self, camera_id: int = 0):
        # Initialize the webcam
        cap = cv2.VideoCapture(camera_id)  # 0 is the default camera

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        return cap

    def capture(self) -> ImageCV:
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to capture frame from camera")
        return frame  # type: ignore


class VideoLoop:
    def __init__(self, video_producer: VideoAbstract):
        self.video_producer = video_producer

    def draw(self, frame: ImageCV):
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

    def process(self, model: Detector):
        while True:
            frame = self.video_producer.capture()

            predictions: list[Prediction] = []
            # predictions = model.predict(model.preprocess(frame))
            frame_with_box = draw_boxes(frame, predictions)
            self.draw(frame_with_box)

    def process_loop(
        self,
        frame_q: Queue[tuple[int, ImageCV]],
        pred_q: Queue[tuple[int, list[Prediction]]],
    ):
        frame_id = -1
        predictions = []
        pred_frame_id = 0
        while True:
            frame_id += 1
            frame = self.video_producer.capture()
            frame_q.put((frame_id, frame))

            try:
                new_pred_frame_id, new_predictions = pred_q.get_nowait()
            except:
                # if queue not ready, just fill with old predictions
                new_pred_frame_id, new_predictions = pred_frame_id, predictions
            pred_frame_id, predictions = new_pred_frame_id, new_predictions
            print(f"current_frame: {frame_id} prediction_frame = {pred_frame_id}")
            frame_with_box = draw_boxes(frame, predictions)
            self.draw(frame_with_box)


def consume_frames(frame_q: Queue[tuple[int, ImageCV]]) -> tuple[int, ImageCV]:
    while not frame_q.empty():
        frame_id, frame = frame_q.get()
        try:
            frame_id, frame = frame_q.get_nowait()
        except:
            break
    return frame_id, frame  # type: ignore


def predict_loop(
    model: Detector,
    frame_q: Queue[tuple[int, ImageCV]],
    pred_q: Queue[tuple[int, list[Prediction]]],
):
    while True:
        frame_id, frame = frame_q.get()
        old_frame_id = frame_id
        while not frame_q.empty():
            try:
                frame_id, frame = frame_q.get_nowait()
            except:
                break
        skipped_frames = frame_id - old_frame_id
        print(f"Skipped frames {skipped_frames}")
        print(f"Working on frame_id={frame_id}")
        predictions = model.predict(model.preprocess(frame))
        pred_q.put(
            (
                frame_id,
                predictions,
            )
        )
        print(f"Predictions on frame_id={frame_id} done. ")
        frame_q.task_done()


def main():
    parser = argparse.ArgumentParser(description="Create a video loop")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to use")
    parser.add_argument(
        "--camera", type=str, default=0, help="camera id or file to use as video"
    )
    args = parser.parse_args()

    input_video = VideoOpenCV(camera_id=args.camera)

    # Classic
    det = Detector(device=args.device)
    obj = VideoLoop(video_producer=input_video)
    # obj.process(det)

    frame_queue: Queue[tuple[int, ImageCV]] = Queue(maxsize=1000)
    predictions_queue: Queue[tuple[int, list[Prediction]]] = Queue(maxsize=1000)

    # daemon=True is for killing the thread if main reach the end or exit
    loop = threading.Thread(
        target=predict_loop,
        args=(det, frame_queue, predictions_queue),
        daemon=True,
    )
    loop.start()
    obj.process_loop(frame_queue, predictions_queue)


if __name__ == "__main__":
    main()
