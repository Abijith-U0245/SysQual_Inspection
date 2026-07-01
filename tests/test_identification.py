import unittest
import numpy as np
import cv2

from test_calibration import identify_part_from_frame


class IdentificationTests(unittest.TestCase):
    def test_identify_part_from_frame_reports_measurement(self):
        frame = np.full((600, 600, 3), 240, dtype=np.uint8)
        cv2.circle(frame, (300, 300), 120, (40, 40, 40), thickness=-1)

        result = identify_part_from_frame(frame, px_per_mm=10.0, nominal_od_mm=24.0)

        self.assertEqual(result["result"], "PASS")
        self.assertGreater(result["od_mm"], 0)
        self.assertGreater(result["width_mm"], 0)
        self.assertGreater(result["height_mm"], 0)


if __name__ == "__main__":
    unittest.main()
