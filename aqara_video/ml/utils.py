from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
from torch import Tensor

from aqara_video.core.types import Image


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Prediction:
    boxes: list[Box]
    scores: list[float]
    labels: list[int]
    categories: list[str]


def draw_box_with_label(
    frame: Image,
    box: Box,
    label: str,
    score: float,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Draw a bounding box with a label on an image.

    Args:
        frame: The image to draw on
        box: The bounding box to draw (Box object with x1, y1, x2, y2 attributes)
        label: The text label to display
        score: The confidence score to display
        box_color: RGB color tuple for the box and label background
        text_color: RGB color tuple for the text

    Returns:
        None: The frame is modified in-place
    """
    text_params = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1,
        "thickness": 2,
    }
    # Access attributes directly from Box object
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

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


def draw_boxes(
    frame: Image, predictions: List[Prediction], threshold: float = 0.5
) -> Image:
    """
    Draw bounding boxes on an image based on object detection predictions.

    Args:
        frame: The image to draw bounding boxes on
        predictions: A list of Prediction objects containing detection results
        threshold: Confidence threshold for filtering detections (0.0 to 1.0)

    Returns:
        Image: The frame with bounding boxes and labels drawn on it
    """
    if len(predictions) == 0:
        print("nothing was detected")
        return frame

    # Get the first prediction from the list
    for prediction in predictions:
        # Draw bounding boxes and labels on the frame
        for i, box in enumerate(prediction.boxes):
            if prediction.scores[i] > threshold:
                # Pass the Box object directly to draw_box_with_label
                draw_box_with_label(
                    frame, box, prediction.categories[i], prediction.scores[i]
                )
    return frame


def to_predictions(
    model_output: List[Dict[str, Tensor]], categories: List[str]
) -> List[Prediction]:
    """
    Convert the model output tensors to a list of strongly-typed Prediction objects.

    Args:
        model_output: The raw output from the model, consisting of a list of dictionaries
                      with tensors for 'boxes', 'labels', and 'scores'.
        categories: A list of category names corresponding to the model's label indices.

    Returns:
        List[Prediction]: A list of Prediction objects, one for each input image.

    Note:
        During inference, the model returns post-processed predictions as a List[Dict[str,Tensor]],
        one for each input image. The fields of the Dict are:
        - boxes (FloatTensor[N, 4]): predicted boxes in [x1, y1, x2, y2] format
        - labels (Int64Tensor[N]): predicted label indices
        - scores (Tensor[N]): confidence scores for each detection
    """
    predictions = []
    for pred in model_output:
        boxes = [[int(z) for z in box] for box in pred["boxes"].tolist()]
        scores = pred["scores"].tolist()
        labels = pred["labels"].tolist()
        categories_list = [categories[label] for label in labels]

        predictions.append(
            Prediction(
                boxes=[Box(*box) for box in boxes],
                scores=scores,
                labels=labels,
                categories=categories_list,
            )
        )
    return predictions
