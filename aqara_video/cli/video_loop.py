import argparse
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue

import cv2
import torch
from PIL import Image
from torchvision import models, transforms


class Detector:
    def __init__(self, device: str = "cpu"):
        weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.labels = list(weights.meta["categories"])

        self.device = torch.device(device)
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights
        )
        self.model.to(self.device)
        self.model.eval()

    def transform(self):
        ret = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert the image to a tensor
            ]
        )
        return ret

    def preprocess(self, cv_image):
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform()(pil_image).to(self.device)
        return image_tensor

    def predict(self, image_tensor):
        def tensor_to_list(pred):
            ans = {k: v.tolist() for k, v in pred.items()}
            ans["categories"] = [self.labels[i] for i in ans["labels"]]
            map_int = lambda x: [int(z) for z in x]
            ans["boxes"] = [map_int(box) for box in ans["boxes"]]
            return ans

        with torch.no_grad():
            predictions = self.model([image_tensor])
            ans = [tensor_to_list(x) for x in predictions]
            return ans


class VideoAbstract(ABC):
    @abstractmethod
    def capture(self):
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

    def capture(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError
        return frame


class VideoLoop:
    def __init__(self, video_producer: VideoAbstract):
        self.video_producer = video_producer

    def _draw_box_with_label(
        self, frame, box, label, score, box_color=(0, 255, 0), text_color=(0, 0, 0)
    ):
        text_params = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 1,
            "thickness": 2,
        }
        x1, y1, x2, y2 = box
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create label text
        label_text = f"{label}: {score:.2f}"

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, **text_params)
        border = 5
        text_width += border
        text_height += border

        # Draw filled rectangle for text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 5),  # Top-left corner
            (x1 + text_width, y1),  # Bottom-right corner
            box_color,  # Green color
            -1,  # Filled rectangle
        )

        # Draw text
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 5),  # Position slightly above box
            text_params["fontFace"],
            text_params["fontScale"],
            text_color,  # Black text color
            text_params["thickness"],
        )

    def draw_boxes(self, frame, predictions, threshold=0.5):
        # frame = self.frame
        # predictions = self.predictions
        if len(predictions) == 0:
            print("nothing was detected")
            return frame
        # Get the predicted bounding boxes, labels, and scores
        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        labels = predictions[0]["categories"]

        # Draw bounding boxes and labels on the frame
        for i, box in enumerate(boxes):
            if scores[i] > threshold:
                self._draw_box_with_label(frame, box, labels[i], scores[i])
        return frame

    def draw(self, frame):
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

    def process(self, model):
        while True:
            frame = self.video_producer.capture()

            predictions = []
            # predictions = model.predict(model.preprocess(frame))
            frame_with_box = self.draw_boxes(frame, predictions)
            self.draw(frame_with_box)

    def process_loop(self, frame_q, pred_q):
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
            frame_with_box = self.draw_boxes(frame, predictions)
            self.draw(frame_with_box)


def consume_frames(frame_q):
    while not frame_q.empty():
        frame_id, frame = frame_q.get()
        try:
            frame_id, frame = frame_q.get_nowait()
        except:
            break
    return frame_id, frame


def predict_loop(model, frame_q, pred_q):
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

    frame_queue = Queue(maxsize=1000)
    predictions_queue = Queue(maxsize=1000)

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
