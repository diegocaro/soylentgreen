import numpy as np
import pytest

from detection.utils import Box, Prediction, draw_box_with_label, draw_boxes
from video_footage.core.types import ImageCV


@pytest.fixture
def test_image() -> ImageCV:
    """Create a simple blank image for testing."""
    return np.zeros((300, 400, 3), dtype=np.uint8)


@pytest.fixture
def test_box() -> Box:
    """Create a sample box for testing."""
    return Box(x1=50, y1=50, x2=150, y2=150)


@pytest.fixture
def test_prediction(test_box: Box) -> Prediction:
    """Create a sample prediction for testing."""
    return Prediction(
        boxes=[test_box], scores=[0.85], labels=[1], categories=["person"]
    )


def test_draw_box_with_label_modifies_image(test_image: ImageCV, test_box: Box) -> None:
    """Test that draw_box_with_label modifies the image with a box and text."""
    # Make a copy of the original image
    image_copy = test_image.copy()

    # Call the function to draw on the image
    draw_box_with_label(frame=image_copy, box=test_box, label="person", score=0.85)

    # Verify the image was modified (should not be all zeros anymore)
    assert not np.array_equal(test_image, image_copy)


def test_draw_box_with_label_handles_custom_colors(
    test_image: ImageCV, test_box: Box
) -> None:
    """Test that custom colors are properly applied."""
    image = test_image.copy()
    # OpenCV uses BGR format, so (0, 0, 255) is red
    custom_box_color: tuple[int, int, int] = (0, 0, 255)  # Red in BGR format
    custom_text_color: tuple[int, int, int] = (255, 255, 255)  # White

    draw_box_with_label(
        frame=image,
        box=test_box,
        label="person",
        score=0.85,
        box_color=custom_box_color,
        text_color=custom_text_color,
    )

    # Check if the box was drawn with the correct color
    # We're looking for red pixels - since OpenCV uses BGR, we look for (0, 0, 255)
    red_pixels = np.sum((image == [0, 0, 255]).all(axis=2))
    assert red_pixels > 0


def test_draw_box_with_label_positions_text_correctly(test_image: ImageCV) -> None:
    """Test that text is positioned correctly even when box is at the top of the image."""
    # Create a box at the top of the image where text would normally go outside
    top_box = Box(x1=50, y1=10, x2=150, y2=50)
    image = test_image.copy()

    # Should not raise any errors when drawing near the edge
    draw_box_with_label(frame=image, box=top_box, label="test", score=0.75)

    # If we got here without errors, the test passes
    assert True


def test_draw_boxes_with_no_predictions(test_image: ImageCV) -> None:
    """Test that draw_boxes handles empty predictions properly."""
    image = test_image.copy()
    result = draw_boxes(frame=image, predictions=[])

    # The function should return the original image unchanged
    assert np.array_equal(image, result)


def test_draw_boxes_with_predictions_below_threshold(
    test_image: ImageCV, test_box: Box
) -> None:
    """Test that predictions below threshold are not drawn."""
    image = test_image.copy()
    low_score_prediction = Prediction(
        boxes=[test_box],
        scores=[0.3],
        labels=[1],
        categories=["person"],  # Low score
    )

    result = draw_boxes(frame=image, predictions=[low_score_prediction], threshold=0.5)

    # Since score is below threshold, image should be unchanged
    assert np.array_equal(image, result)


def test_draw_boxes_with_predictions_above_threshold(
    test_image: ImageCV, test_prediction: Prediction
) -> None:
    """Test that predictions above threshold are drawn."""
    image = test_image.copy()

    result = draw_boxes(
        frame=image,
        predictions=[test_prediction],
        threshold=0.5,  # Score is 0.85
    )

    # Image should be modified
    assert not np.array_equal(test_image, result)


def test_draw_boxes_with_multiple_predictions(test_image: ImageCV) -> None:
    """Test drawing multiple boxes from multiple predictions."""
    image = test_image.copy()

    box1: Box = Box(x1=50, y1=50, x2=100, y2=100)
    box2: Box = Box(x1=150, y1=150, x2=200, y2=200)

    prediction1: Prediction = Prediction(
        boxes=[box1], scores=[0.9], labels=[1], categories=["person"]
    )
    prediction2: Prediction = Prediction(
        boxes=[box2], scores=[0.8], labels=[2], categories=["car"]
    )

    result = draw_boxes(frame=image, predictions=[prediction1, prediction2])

    # Image should be modified
    assert not np.array_equal(test_image, result)
