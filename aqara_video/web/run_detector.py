import logging

from aqara_video.web.detector import YellowBoxDetector

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    yellow_box_detector = YellowBoxDetector()
    pass
