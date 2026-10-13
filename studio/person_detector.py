"""Optional CPU-only worker, run with a configured Torchvision Python.

Inputs/results/weights stay under ignored application state. The fixed public
COCO weights are downloaded on the first explicitly requested person check.
No identity matching, face embeddings, or cloud image requests are performed.
"""
import json
from pathlib import Path
import sys


def assess_boxes(detection):
    """Expose nested detections as a count range, not proof of a second person.

    COCO person detections can cover a head separately from its own body.
    Geometry alone cannot distinguish this from two overlapping people, so
    retain every box and require review instead of silently suppressing one.
    """
    if detection.get("status") != "complete" or "boxes" not in detection:
        return dict(detection)
    threshold = detection.get("threshold", .5)
    boxes = detection.get("boxes", [])
    active = [i for i, b in enumerate(boxes) if b["score"] >= threshold]
    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    ambiguous = []
    for i in active:
        small = boxes[i]["box"]
        sa = area(small)
        if not sa:
            continue
        for j in active:
            if i == j:
                continue
            large = boxes[j]["box"]
            la = area(large)
            intersection = [max(small[0], large[0]), max(small[1], large[1]),
                            min(small[2], large[2]), min(small[3], large[3])]
            if sa <= la * .3 and area(intersection) / sa >= .75:
                ambiguous.append(i)
                break
    maximum = len(active)
    return {**detection, "assessment_version": 2, "raw_count": maximum,
            "count": None if ambiguous else maximum,
            "count_min": maximum - len(ambiguous), "count_max": maximum,
            "overlap_candidates": ambiguous,
            "uncertain": bool(ambiguous) or any(b["score"] < threshold for b in boxes)}


def main():
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
    from PIL import Image, ImageOps
    torch.set_num_threads(2)
    request_path, output_path, model_dir = map(Path, sys.argv[1:4])
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(model_dir),
                                             map_location="cpu", progress=False, check_hash=True, weights_only=True)
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91).eval()
    model.load_state_dict(state)
    transform = weights.transforms()
    with output_path.open("w", encoding="utf-8") as output:
        for item in json.loads(request_path.read_text(encoding="utf-8")):
            try:
                with Image.open(item["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                # Limit decoding-to-tensor work; detector itself uses a 320px input.
                image.thumbnail((1280, 1280))
                with torch.inference_mode():
                    result = model([transform(image)])[0]
                boxes = [{"box": [round(float(v) / (image.width if i % 2 == 0 else image.height), 5)
                                   for i, v in enumerate(box)], "score": round(float(score), 4)}
                         for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
                         if int(label) == 1 and float(score) >= .25]
                detection = assess_boxes({"status": "complete", "model": "ssdlite320_mobilenet_v3_large/COCO_V1",
                             "threshold": .5, "count": sum(b["score"] >= .5 for b in boxes), "boxes": boxes,
                             "uncertain": any(b["score"] < .5 for b in boxes), "source_size": [width, height]})
            except Exception:
                detection = {"status": "failed", "reason": "Could not inspect this image; review it manually."}
            output.write(json.dumps({"id": item["id"], "detection": detection}) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
