"""Synthetic box geometry; no model, private images or local services."""
import unittest
from studio.person_detector import assess_boxes


def detection(*boxes):
    return {"status": "complete", "threshold": .5,
            "boxes": [{"box": box, "score": score} for box, score in boxes]}


class PersonBoxesCase(unittest.TestCase):
    def test_nested_box_is_ambiguous_not_a_confirmed_extra_person(self):
        value = detection(([0, 0, 1, 1], .99), ([.3, .02, .6, .3], .91))
        result = assess_boxes(value)
        self.assertIsNone(result["count"])
        self.assertEqual((result["count_min"], result["count_max"]), (1, 2))
        self.assertEqual(result["boxes"], value["boxes"])
        self.assertEqual(result["overlap_candidates"], [1])
        self.assertEqual(assess_boxes(result), result)

    def test_separate_people_and_similar_size_overlap_remain_two(self):
        for second in ([.55, .05, .95, .95], [.25, .05, .65, .95]):
            result = assess_boxes(detection(([.05, .05, .45, .95], .96), (second, .93)))
            self.assertEqual(result["count"], 2)
            self.assertFalse(result["uncertain"])

    def test_weak_candidate_is_retained_but_not_counted(self):
        result = assess_boxes(detection(([0, 0, .45, 1], .9), ([.5, 0, 1, 1], .3)))
        self.assertEqual(result["count"], 1)
        self.assertTrue(result["uncertain"])
        self.assertEqual(len(result["boxes"]), 2)

    def test_empty_or_failed_detection(self):
        self.assertEqual(assess_boxes(detection())["count"], 0)
        self.assertEqual(assess_boxes({"status": "failed"}), {"status": "failed"})
