import cv2


def draw_box_with_label(
    frame, box, label, score, box_color=(0, 255, 0), text_color=(0, 0, 0)
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


def draw_boxes(frame, predictions, threshold=0.5):
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
            draw_box_with_label(frame, box, labels[i], scores[i])
    return frame
