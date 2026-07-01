import unittest
import numpy as np
import cv2

from test_calibration import measure_part


class MeasurePartTests(unittest.TestCase):
    def test_detects_dark_circle_on_light_background(self):
        frame = np.full((600, 600, 3), 240, dtype=np.uint8)
        cv2.circle(frame, (300, 300), 120, (40, 40, 40), thickness=-1)

        od_mm, annotated, width_mm, height_mm = measure_part(frame, px_per_mm=10.0)

        self.assertIsNotNone(od_mm)
        self.assertGreater(od_mm, 0)
        self.assertGreater(width_mm, 0)
        self.assertGreater(height_mm, 0)


if __name__ == "__main__":
    unittest.main()
