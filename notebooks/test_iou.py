import unittest
import math

import torch
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import affine_transform

from detectron2._C import box_iou_rotated

def calculate_iou(box1, box2, ariou=False):
    """
    Calculate IoU for rotated boxes in a right-handed coordinate system, with angles in degrees CCW from the x-axis.

    Each box is specified as (x_center, y_center, width, height, angle).
    """

    # Convert angles to radians
    box1_a_rad = math.radians(box1[4])
    box2_a_rad = math.radians(box2[4])
    
    scale = max(0, math.cos(box1_a_rad - box2_a_rad))
    
    def get_box_polygon(box):
        cx, cy, w, h, angle = box
        # Define the corners of the box before rotation
        corners = [(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2), (-w / 2, -h / 2)]
        
        # Convert angle from degrees to radians and apply right-handed (CCW) rotation
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # Rotation matrix components for affine transformation
        a, b, c, d, e, f = cos_a, sin_a, -sin_a, cos_a, cx, cy

        # Apply affine transformation: rotation and translation
        rotated_polygon = Polygon([affine_transform(Point(x, y), [a, b, c, d, e, f]) for x, y in corners])
        return rotated_polygon

    poly1 = get_box_polygon(box1)
    poly2 = get_box_polygon(box2)

    # Calculate intersection and union areas
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    # Compute IoU
    iou = intersection / union if union != 0 else 0

    return iou * scale if ariou else iou

class TestBoxIoURotated(unittest.TestCase):

    def test_exact_overlap_various_angles(self):
        base_box = torch.FloatTensor([100, 100, 100, 50, 0])  # Base box
        angles = torch.arange(0, 360, 30)  # Test every 30 degrees
        for angle in angles:
            with self.subTest(angle=angle.item()):
                test_box = torch.FloatTensor([100, 100, 100, 50, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0), test_box.unsqueeze(0), False)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle.item()}")

    def test_large_angle_differences(self):
        base_box = torch.FloatTensor([100, 100, 100, 100, 0])  # Base box
        angles = torch.FloatTensor([181, 270, 359])  # Large angle differences
        for angle in angles:
            with self.subTest(angle=angle.item()):
                # Rotate test_box by a large angle difference
                test_box = torch.FloatTensor([100, 100, 100, 100, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0), test_box.unsqueeze(0), False)
                # The expected IoU needs to be calculated based on the geometry
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())  # Placeholder for expected IoU; calculate based on actual geometry
                self.assertAlmostEqual(iou.item(), shapely_iou, places=2, msg=f"Failed at angle {angle.item()}")

    def test_edge_cases_thin_rectangles(self):
        base_box = torch.FloatTensor([100, 100, 200, 10, 0])  # Very thin rectangle
        angles = torch.FloatTensor([10, 45, 80, 100, 160, 200, 250, 290, 350])
        for angle in angles:
            with self.subTest(angle=angle.item()):
                test_box = torch.FloatTensor([100, 100, 200, 10, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0), test_box.unsqueeze(0), False)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy()) 
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle.item()}")

    def test_no_overlap(self):
        base_box = torch.FloatTensor([50, 50, 100, 100, 0])
        test_box = torch.FloatTensor([250, 250, 100, 100, 0])  # Far apart, no overlap
        iou = box_iou_rotated(base_box.unsqueeze(0), test_box.unsqueeze(0), False)
        shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())
        self.assertAlmostEqual(iou.item(), shapely_iou, places=4)

    def test_random_rotations(self):
        base_box = torch.FloatTensor([100, 100, 150, 100, 0])
        for _ in range(100):  # 10 random tests
            angle = np.random.randint(0, 360)
            with self.subTest(angle=angle):
                test_box = torch.FloatTensor([120, 120, 150, 100, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0), test_box.unsqueeze(0), False)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle} base_box={base_box} test_box={test_box}")   

    def test_random_rotations_ariou(self):
        base_box = torch.FloatTensor([100, 100, 150, 100, 0])
        for _ in range(100):  # 10 random tests
            angle = np.random.randint(0, 360)
            with self.subTest(angle=angle):
                test_box = torch.FloatTensor([120, 120, 150, 100, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0), test_box.unsqueeze(0), True)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy(), ariou=True)
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle} base_box={base_box} test_box={test_box}")

class TestBoxIoURotatedCUDA(unittest.TestCase):
    def test_exact_overlap_various_angles(self):
        base_box = torch.FloatTensor([100, 100, 100, 50, 0])  # Base box
        angles = torch.arange(0, 360, 30)  # Test every 30 degrees
        for angle in angles:
            with self.subTest(angle=angle.item()):
                test_box = torch.FloatTensor([100, 100, 100, 50, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0).cuda(), test_box.unsqueeze(0).cuda(), False)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle.item()}")

    def test_large_angle_differences(self):
        base_box = torch.FloatTensor([100, 100, 100, 100, 0])  # Base box
        angles = torch.FloatTensor([181, 270, 359])  # Large angle differences
        for angle in angles:
            with self.subTest(angle=angle.item()):
                # Rotate test_box by a large angle difference
                test_box = torch.FloatTensor([100, 100, 100, 100, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0).cuda(), test_box.unsqueeze(0).cuda(), False)
                # The expected IoU needs to be calculated based on the geometry
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())  # Placeholder for expected IoU; calculate based on actual geometry
                self.assertAlmostEqual(iou.item(), shapely_iou, places=2, msg=f"Failed at angle {angle.item()}")

    def test_edge_cases_thin_rectangles(self):
        base_box = torch.FloatTensor([100, 100, 200, 10, 0])  # Very thin rectangle
        angles = torch.FloatTensor([10, 45, 80, 100, 160, 200, 250, 290, 350])
        for angle in angles:
            with self.subTest(angle=angle.item()):
                test_box = torch.FloatTensor([100, 100, 200, 10, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0).cuda(), test_box.unsqueeze(0).cuda(), False)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy()) 
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle.item()}")

    def test_no_overlap(self):
        base_box = torch.FloatTensor([50, 50, 100, 100, 0])
        test_box = torch.FloatTensor([250, 250, 100, 100, 0])  # Far apart, no overlap
        iou = box_iou_rotated(base_box.unsqueeze(0).cuda(), test_box.unsqueeze(0).cuda(), False)
        shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())
        self.assertAlmostEqual(iou.item(), shapely_iou, places=4)

    def test_random_rotations(self):
        base_box = torch.FloatTensor([100, 100, 150, 100, 0])
        for _ in range(100):  # 10 random tests
            angle = np.random.randint(0, 360)
            with self.subTest(angle=angle):
                test_box = torch.FloatTensor([120, 120, 150, 100, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0).cuda(), test_box.unsqueeze(0).cuda(), False)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy())
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle} base_box={base_box} test_box={test_box}")   

    def test_random_rotations_ariou(self):
        base_box = torch.FloatTensor([100, 100, 150, 100, 0])
        for _ in range(100):  # 10 random tests
            angle = np.random.randint(0, 360)
            with self.subTest(angle=angle):
                test_box = torch.FloatTensor([120, 120, 150, 100, angle])
                iou = box_iou_rotated(base_box.unsqueeze(0).cuda(), test_box.unsqueeze(0).cuda(), True)
                shapely_iou = calculate_iou(base_box.numpy(), test_box.numpy(), ariou=True)
                self.assertAlmostEqual(iou.item(), shapely_iou, places=4, msg=f"Failed at angle {angle} base_box={base_box} test_box={test_box}")

if __name__ == '__main__':
    unittest.main()
