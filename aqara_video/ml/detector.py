from typing import Any

import torch
from torchvision import models, transforms

from aqara_video.core.images import bgr_to_rgb
from aqara_video.core.types import ImageCV
from aqara_video.ml.utils import Prediction, to_predictions


class Detector:
    """
    A class for object detection using a Faster R-CNN model with MobileNet backbone.

    This class handles initialization of the PyTorch model, preprocessing of input images,
    and prediction of object bounding boxes, labels, and confidence scores.

    Features:
    - Batch processing for faster inference
    - Optimized preprocessing pipeline
    - Compatible with both single image and batch processing workflows
    """

    def __init__(self, device: str = "cpu", batch_size: int = 4):
        """
        Initialize the detector with a pre-trained Faster R-CNN model.

        Args:
            device: The device to run the model on ('cpu' or 'cuda')
            batch_size: Number of frames to process in a single batch (default: 4)
        """
        weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.labels = list(weights.meta["categories"])

        self.device = torch.device(device)
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights
        )
        self.model.to(self.device)
        self.model.eval()

        # Batch processing parameters
        self.batch_size = batch_size
        # Pre-allocate the transform once
        self._transform = transforms.Compose([transforms.ToTensor()])
        self.frame_buffer: list[ImageCV] = []
        self.frame_ids_buffer: list[int] = []

    def transform(self) -> transforms.Compose:
        """Return the image transformation pipeline for preprocessing (legacy method)."""
        return self._transform

    def preprocess(self, image: ImageCV) -> torch.Tensor:
        """
        Preprocess a single image for model input (legacy method).

        Args:
            image: A BGR format image (e.g., from OpenCV)

        Returns:
            A tensor ready for model input
        """
        # Use the optimized preprocessing approach
        rgb_frame = bgr_to_rgb(image)
        image_tensor = self._transform(rgb_frame).to(self.device)
        return image_tensor

    def preprocess_batch(self, frames: list[ImageCV]) -> list[torch.Tensor]:
        """
        Process multiple frames at once.

        Args:
            frames: List of BGR format images (e.g., from OpenCV)

        Returns:
            List of tensor representations of the input frames
        """
        tensors = []
        for frame in frames:
            # Convert BGR to RGB using our centralized function
            rgb_frame = bgr_to_rgb(frame)
            # Convert to tensor
            tensor = self._transform(rgb_frame).to(self.device)
            tensors.append(tensor)
        return tensors

    def predict(self, image_tensor: torch.Tensor) -> list[Prediction]:
        """
        Make object detection predictions on a single preprocessed image tensor.
        Legacy method for backward compatibility.

        Args:
            image_tensor: A preprocessed image tensor

        Returns:
            A list of dictionaries containing detection results with keys:
            - boxes: list of [x1, y1, x2, y2] coordinates
            - labels: list of class indices
            - scores: list of confidence scores
            - categories: list of class names
        """
        # This method now uses the batched prediction with batch size of 1
        # for backward compatibility
        with torch.no_grad():
            predictions = self.model([image_tensor])

            return to_predictions(predictions, self.labels)

    def predict_batch(
        self, frame_ids: list[int], frames: list[ImageCV]
    ) -> list[tuple[int, list[dict[str, Any]]]]:
        """
        Predict on a batch of frames.

        Args:
            frame_ids: List of identifiers for each frame
            frames: List of BGR format images

        Returns:
            List of (frame_id, predictions) tuples
        """
        # TODO: convert to Prediction objects
        # Preprocess frames into tensors
        tensors = self.preprocess_batch(frames)

        # Predict on the batch of tensors
        with torch.no_grad():
            predictions = self.model(tensors)

        # Convert to list format
        results = []
        for i, pred in enumerate(predictions):
            processed_pred = {k: v.tolist() for k, v in pred.items()}
            processed_pred["categories"] = [
                self.labels[i] for i in processed_pred["labels"]
            ]
            processed_pred["boxes"] = [
                [int(z) for z in box] for box in processed_pred["boxes"]
            ]
            results.append((frame_ids[i], [processed_pred]))

        return results

    def add_to_batch(self, frame_id: int, frame: ImageCV) -> list[Prediction]:
        """
        Add a frame to the batch buffer.

        Args:
            frame_id: Identifier for the frame
            frame: The BGR format image

        Returns:
            List of (frame_id, predictions) tuples if batch is full and processed,
            empty list otherwise
        """
        self.frame_buffer.append(frame)
        self.frame_ids_buffer.append(frame_id)

        # If we've reached batch size, process the batch
        if len(self.frame_buffer) >= self.batch_size:
            results = self.predict_batch(self.frame_ids_buffer, self.frame_buffer)
            # Clear buffers
            self.frame_buffer = []
            self.frame_ids_buffer = []
            return results
        return []

    def flush_batch(self) -> list[Prediction]:
        """
        Process any remaining frames in the buffer.

        Returns:
            List of (frame_id, predictions) tuples
        """
        if not self.frame_buffer:
            return []

        results = self.predict_batch(self.frame_ids_buffer, self.frame_buffer)
        # Clear buffers
        self.frame_buffer = []
        self.frame_ids_buffer = []
        return results
