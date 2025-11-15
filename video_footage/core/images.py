import cv2
import numpy as np

from video_footage.core.types import ImageCV


def encode_to_jpeg(frame: ImageCV, quality: int = 95) -> bytes:
    """
    Encode an image frame to JPEG format.

    Args:
        frame (np.ndarray): The image frame to encode, in BGR format as used by OpenCV
        quality (int, optional): JPEG quality from 0 to 100. Defaults to 95.

    Returns:
        bytes: The JPEG encoded image as bytes
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, jpeg = cv2.imencode(".jpg", frame, encode_params)
    return jpeg.tobytes()


def bgr_to_rgb(image: ImageCV) -> np.ndarray:
    """
    Convert an image from BGR to RGB color space.

    Args:
        image (np.ndarray): The image in BGR format (as used by OpenCV)

    Returns:
        np.ndarray: The image in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
