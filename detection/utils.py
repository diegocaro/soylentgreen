from dataclasses import dataclass

import cv2
from torch import Tensor

from video_footage.core.types import ImageCV


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
    frame: ImageCV,
    box: Box,
    label: str,
    score: float,
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (0, 0, 0),
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
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Create label text
    label_text = f"{label}: {score:.2f}"

    # Get text size
    (text_width, text_height), _baseline = cv2.getTextSize(label_text, **text_params)
    border = 5
    text_width += border
    text_height += border

    padding = 5
    # Draw filled rectangle for text background
    top_left = (x1, y1 - text_height - padding)
    if top_left[1] < text_height:
        # If the rectangle goes above the image, adjust the position
        top_left = (x1, y1 + text_height + padding)

    cv2.rectangle(
        frame,
        top_left,  # Top-left corner
        (x1 + text_width, y1),  # Bottom-right corner
        box_color,
        -1,  # Filled rectangle
    )

    top_left_text = (x1, y1 - padding)
    if top_left_text[1] < text_height:
        # If the text goes above the image, adjust the position
        top_left_text = (x1, y1 + text_height)

    # Draw text
    cv2.putText(
        frame,
        label_text,
        top_left_text,  # Position slightly above box
        text_params["fontFace"],
        text_params["fontScale"],
        text_color,
        text_params["thickness"],
    )


def draw_boxes(
    frame: ImageCV, predictions: list[Prediction], threshold: float = 0.5
) -> ImageCV:
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
        print("Nothing was detected")
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
    model_output: list[dict[str, Tensor]], categories: list[str]
) -> list[Prediction]:
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

        # Validate label indices are within range of categories
        categories_list = []
        for label in labels:
            if 0 <= label < len(categories):
                categories_list.append(categories[label])
            else:
                print(f"Label index {label} out of range. Using 'unknown' as category.")
                categories_list.append("unknown")

        predictions.append(
            Prediction(
                boxes=[Box(*box) for box in boxes],
                scores=scores,
                labels=labels,
                categories=categories_list,
            )
        )
    return predictions
