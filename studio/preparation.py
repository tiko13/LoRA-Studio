"""Deterministic image preparation shared by preview, captioning and export.

Bucket selection follows AI Toolkit toolkit/buckets.py (area-capped grid).
Original bytes are never modified. All geometry uses EXIF-oriented pixels.
"""
import hashlib
import io
import json
import math
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from .person_detector import assess_boxes
from .categories import CATEGORIES
from . import object_preparation as objects

VERSION = 1
COVERAGE = {
    "framing": ["close-up", "upper-body", "full-body", "detail", "other", "uncertain"],
    "viewpoint": ["front", "three-quarter", "profile", "back", "other", "uncertain"],
    "lighting": ["daylight", "studio", "indoor", "low-light", "mixed", "uncertain"],
}


def coverage_schema(kind='character'):
    return COVERAGE if kind == 'character' else CATEGORIES[kind]['coverage']


def normalize_coverage(value, kind='character'):
    categories = coverage_schema(kind)
    if not isinstance(value, dict) or any(k not in categories or v not in categories[k] for k, v in value.items()):
        raise ValueError("Coverage tags must use the supported categories")
    return dict(value)


def effective_coverage(project, item):
    kind = project['kind']
    # Legacy untyped tags and overrides belong to Person. Never reinterpret them.
    manual = (item.get('coverage_override', {}) if kind == 'character' else
              item.get('coverage_overrides', {}).get(kind, {}))
    suggested = item.get('coverage', {}) if item.get('coverage_kind', 'character') == kind and not item.get('preparation_stale') else {}
    if kind == 'concept' and item.get('coverage_intent_digest'):
        from .domain import intent_digest
        if item['coverage_intent_digest'] != intent_digest(project):
            suggested = {}
    categories = coverage_schema(kind)
    manual = {k: v for k, v in manual.items() if k in categories and v in categories[k]}
    suggested = {k: v for k, v in suggested.items() if k in categories and v in categories[k]}
    return {**suggested, **manual}, manual


def category_advisories(project, item, geometry):
    kind = project['kind']
    if kind == 'character':
        return []
    tags, _ = effective_coverage(project, item)
    flags = []
    checks = {
        'object': ('target_visibility', {'partial', 'obscured', 'absent'}, 'Object visibility needs review; confirm the intended target and defining parts are usable.'),
        'style': ('surface_detail', {'too-small'}, 'Style detail may be too small; inspect texture and linework at the actual training size.'),
        'concept': ('relationship', {'partial', 'unclear', 'absent'}, 'Concept evidence needs review; confirm the selected pose, expression, angle or other target is visible rather than inferred.'),
    }
    key, concerning, message = checks[kind]
    if tags.get(key) in concerning:
        flags.append(message)
    if kind == 'style' and tags.get('border') == 'present':
        flags.append('A visible border was reported. Check whether it belongs to the intended style and survives the training crop. Caption omission does not remove it.')
    if kind == 'style' and project.get('style_family') == 'pixel':
        flags.append('Inspect pixel edges in the training preview: resizing may soften the grid. The source image is preserved.')
    if not tags or tags.get(key, 'uncertain') == 'uncertain':
        flags.append('Category coverage is unknown. Review the image or prepare coverage suggestions; this does not block training.')
    if geometry['removed_percent'] > 1:
        flags.append(CATEGORIES[kind]['crop_note'])
    return flags


def dataset_advisories(project, images):
    """Coverage summaries are evidence-limited advisories, never exclusion rules."""
    if project['kind'] == 'character':
        return []
    included = [i for i in images if i.get('included', True)]
    flags = []
    if project['kind'] == 'style' and project['model'] == 'krea2' and project['trigger'].startswith('lora_'):
        flags.append('Krea 2 style guidance recommends a descriptive trigger phrase. Review the generated trigger under learning rules; it will not be changed automatically.')
    for field, (attribute, message) in CATEGORIES[project['kind']]['diversity'].items():
        if project.get('choices', {}).get(attribute, 'auto') == 'keep':
            continue
        values = [effective_coverage(project, i)[0].get(field, 'uncertain') for i in included]
        known = [v for v in values if v != 'uncertain']
        if len(known) >= 2 and len(set(known)) == 1:
            flags.append(message + f' Based on {len(known)} of {len(included)} included images; unreviewed model tags may be wrong.')
    return flags

PROFILES = {
    "krea2": {"default": 1024, "multiple": 16, "maximum": 2048,
              "note": "Train on Raw; evaluate on Turbo. 1024 is the recommended starting target. Higher targets are experimental."},
    "qwen21": {"default": 1024, "multiple": 32, "maximum": 2048,
               "note": "Start LoRA training at 1024. Native 2K generation does not require 2K training; higher targets need a memory and quality comparison."},
    "flux": {"default": 768, "multiple": 16, "maximum": 1536, "note": "Starter profile; compare checkpoints."},
    "sdxl": {"default": 768, "multiple": 64, "maximum": 1536, "note": "Starter profile; compare checkpoints."},
}


def open_image(source):
    try:
        with Image.open(source) as raw:
            if raw.width * raw.height > 100_000_000 or min(raw.size) < 16:
                raise ValueError("Image dimensions must be at least 16 px and at most 100 megapixels")
            image = ImageOps.exif_transpose(raw)
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image could not be decoded safely; re-export it as PNG or JPEG") from exc


def metadata(source):
    image = open_image(source)
    # A perceptual signature is only a duplicate candidate, never an identity test.
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(gray.get_flattened_data()) if hasattr(gray, "get_flattened_data") else list(gray.getdata())
    bits = sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))
    has_alpha = ("A" in image.getbands() or "transparency" in image.info) and image.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return {"width": image.width, "height": image.height, "dhash": f"{bits:016x}", "has_alpha": has_alpha}


def normalize_options(value=None, inherit=False):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError("Preparation settings must be an object")
    mode = value.get("mode", "auto" if inherit else "crop")
    if mode not in ({"auto", "crop", "pad"} if inherit else {"crop", "pad"}):
        raise ValueError("Choose aspect-ratio crop or padding")
    result = {"mode": mode}
    for key in ("x", "y"):
        raw = value.get(key, .5)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or not 0 <= raw <= 1:
            raise ValueError("Crop position must be between 0 and 1")
        result[key] = raw
    rect = value.get('crop_rect')
    if rect is not None:
        if not inherit:
            raise ValueError('Select crops for individual images, not the project default')
        if (not isinstance(rect, list) or len(rect) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) or not 0 <= v <= 1 for v in rect) or
                rect[2] <= rect[0] or rect[3] <= rect[1]):
            raise ValueError('Crop rectangle must have ordered left, top, right and bottom coordinates between 0 and 1')
        result['crop_rect'] = list(rect)
    aspect = value.get('crop_aspect', 0)
    if isinstance(aspect, bool) or not isinstance(aspect, (int, float)) or not math.isfinite(aspect) or not 0 <= aspect <= 10000:
        raise ValueError('Crop aspect ratio must be a finite non-negative number')
    if aspect:
        if not inherit:
            raise ValueError('Crop aspect ratios apply to individual images')
        result['crop_aspect'] = aspect
    strokes = value.get('hide_mask', [])
    if not isinstance(strokes, list) or len(strokes) > 300:
        raise ValueError('Mask supports up to 300 strokes; clear or simplify the mask')
    if strokes and not inherit:
        raise ValueError('Masks apply to individual images')
    if strokes:
        cleaned = []
        for stroke in strokes:
            if not isinstance(stroke, dict) or stroke.get('tool') not in {'brush', 'rectangle'} or not isinstance(stroke.get('hide'), bool):
                raise ValueError('Invalid hide mask tool')
            size, points = stroke.get('size'), stroke.get('points')
            if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or not .001 <= size <= .5:
                raise ValueError('Mask brush size must be between 0.1% and 50%')
            if (not isinstance(points, list) or not 1 <= len(points) <= 1000 or
                    any(not isinstance(p, list) or len(p) != 2 or
                        any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in points)):
                raise ValueError('Invalid mask points')
            if stroke['tool'] == 'rectangle' and len(points) != 2:
                raise ValueError('A mask rectangle needs two corners')
            cleaned.append({'tool': stroke['tool'], 'hide': stroke['hide'], 'size': size, 'points': points})
        result['hide_mask'] = cleaned
    return result


def bucket(width, height, resolution, multiple):
    target = min(width * height, resolution * resolution)
    scale = math.sqrt(target / (width * height))
    w, h = width * scale / multiple, height * scale / multiple
    candidates = [(a(w) * multiple, b(h) * multiple)
                  for a, b in ((math.floor, math.floor), (math.floor, math.ceil),
                               (math.ceil, math.floor), (math.ceil, math.ceil))]
    candidates = [(w, h) for w, h in candidates if w > 0 and h > 0 and w * h <= resolution ** 2]
    if not candidates:
        # Extremely thin images can round one axis to zero. Keep even these
        # within the budget instead of allocating an unbounded long edge.
        fw, fh = max(multiple, math.floor(w) * multiple), max(multiple, math.floor(h) * multiple)
        if fw >= fh:
            fw = max(multiple, (resolution ** 2 // fh // multiple) * multiple)
        else:
            fh = max(multiple, (resolution ** 2 // fw // multiple) * multiple)
        candidates = [(fw, fh)]
    return min(candidates, key=lambda size: abs(size[0] * size[1] - target))


def plan(project, item):
    width, height = item["width"], item["height"]
    source_width, source_height = width, height
    options = normalize_options(item.get("preparation"), inherit=True)
    region = objects.region(project, item) if project['kind'] == 'object' else None
    custom_crop = options.get('crop_rect')
    if custom_crop:
        # Coordinates refer to the oriented original, including for Object masks.
        l, t, r, b = custom_crop
        left, top = min(width - 1, round(l * width)), min(height - 1, round(t * height))
        region = [left, top, min(width, max(left + 1, round(r * width))),
                  min(height, max(top + 1, round(b * height)))]
    if region:
        width, height = region[2] - region[0], region[3] - region[1]
    profile = PROFILES[project["model"]]
    resolution = project["training"]["resolution"]
    mode = options["mode"]
    if mode == "auto":
        mode = normalize_options(project.get("preparation"))["mode"]
    bw, bh = bucket(width, height, resolution, profile["multiple"])
    if region and project['kind'] == 'object' and not custom_crop and mode == 'crop' and options['mode'] == 'auto':
        # Grid alignment must not shave a protruding part off a tight target box.
        trial_scale = max(bw / width, bh / height)
        trial_w, trial_h = math.ceil(width * trial_scale), math.ceil(height * trial_scale)
        lx, ty = int((trial_w-bw)*options['x']), int((trial_h-bh)*options['y'])
        visible = [region[0]+lx*width/trial_w, region[1]+ty*height/trial_h,
                   region[0]+(lx+bw)*width/trial_w, region[1]+(ty+bh)*height/trial_h]
        target = objects.selection(project, item)['box']
        if (visible[0] > target[0]*source_width or visible[1] > target[1]*source_height or
                visible[2] < target[2]*source_width or visible[3] < target[3]*source_height):
            mode = 'pad'
    scale = (max if mode == "crop" else min)(bw / width, bh / height)
    sw, sh = ((math.ceil(width * scale), math.ceil(height * scale)) if mode == "crop"
              else (max(1, math.floor(width * scale)), max(1, math.floor(height * scale))))
    left = int((sw - bw) * options["x"]) if mode == "crop" else (bw - sw) // 2
    top = int((sh - bh) * options["y"]) if mode == "crop" else (bh - sh) // 2
    removed = 1 - bw * bh / (sw * sh) if mode == "crop" else 0
    result = {"version": VERSION, "width": bw, "height": bh, "source_width": source_width, "source_height": source_height,
              "resize_width": sw, "resize_height": sh, "left": left, "top": top, "mode": mode,
              "removed_percent": round(max(0, removed) * 100, 2), "scale": scale,
              "multiple": profile["multiple"], "rgba": project["model"] == "qwen21"}
    if region:
        result['source_region'] = region
    if project['kind'] == 'object' and region:
        result['object_box'] = objects.selection(project, item)['box']
        result['object_target_signature'] = objects.signature(project, item)
    if custom_crop:
        result['custom_crop'] = True
    if options.get('hide_mask'):
        result['hide_mask'] = options['hide_mask']
    result["digest"] = hashlib.sha256(json.dumps({"source": item["digest"], **result}, sort_keys=True).encode()).hexdigest()
    return result


def render(source, geometry):
    image = open_image(source).convert("RGBA")
    if geometry.get('hide_mask'):
        # A reversible content mask, distinct from Object training loss masks.
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        for stroke in geometry['hide_mask']:
            points = [(round(x * (image.width - 1)), round(y * (image.height - 1))) for x, y in stroke['points']]
            fill = 255 if stroke['hide'] else 0
            if stroke['tool'] == 'rectangle':
                a, b = points
                draw.rectangle((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), fill=fill)
            else:
                width = max(1, round(stroke['size'] * min(image.size)))
                if len(points) > 1:
                    draw.line(points, fill=fill, width=width, joint='curve')
                radius = (width - 1) / 2
                for x, y in points:
                    if width == 1:
                        draw.point((x, y), fill=fill)
                    else:
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)
        image.paste((128, 128, 128, 255), (0, 0, image.width, image.height), mask)
    if geometry.get('source_region'):
        image = image.crop(tuple(geometry['source_region']))
    image = image.resize((geometry["resize_width"], geometry["resize_height"]), Image.Resampling.BICUBIC)
    x, y, w, h = (geometry[k] for k in ("left", "top", "width", "height"))
    if geometry["mode"] == "crop":
        image = image.crop((x, y, x + w, y + h))
    else:
        canvas = Image.new("RGBA", (w, h), (128, 128, 128, 255))
        canvas.paste(image, (x, y))
        image = canvas
    if not geometry["rgba"]:
        background = Image.new("RGBA", image.size, (128, 128, 128, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    return image


def png_bytes(source, geometry):
    buffer = io.BytesIO()
    render(source, geometry).save(buffer, format="PNG")
    return buffer.getvalue()


def advisory(item, geometry, kind='character'):
    flags = []
    if item["width"] * item["height"] < 512 ** 2:
        flags.append("Small source: inspect usable subject detail; resizing cannot recover detail.")
    if geometry["scale"] > 1.02:
        flags.append("Grid alignment slightly enlarges this image.")
    if geometry["removed_percent"] > 1:
        flags.append(f"Training crop removes {geometry['removed_percent']}% of the image area. Check the edges.")
    if geometry["mode"] == "pad":
        flags.append("Padding remains visible; its loss is weighted by the selected Object mask." if kind == 'object' and objects.options(item.get('object_target'))['mask'] != 'none' else "Padding becomes training content; no loss mask is applied.")
    if geometry.get('source_region'):
        r = geometry['source_region']
        if (r[2]-r[0])*(r[3]-r[1]) < 512**2 and item['width']*item['height'] >= 512**2:
            flags.append('Small selected region: inspect detail in the training crop; a large original does not guarantee enough target pixels.')
    if geometry.get('hide_mask'):
        flags.append('Hidden areas are filled with neutral grey in captions and training images; this is not background reconstruction or a loss mask.')
    if geometry.get('custom_crop') and geometry.get('object_box'):
        l, t, r, b = geometry['source_region']
        target = geometry['object_box']
        if (l > target[0]*item['width'] or t > target[1]*item['height'] or
                r < target[2]*item['width'] or b < target[3]*item['height']):
            flags.append('The manual crop cuts into the selected object. Review the target and caption before training.')
    if max(item["width"], item["height"]) / min(item["width"], item["height"]) > 3:
        flags.append("Extreme aspect ratio: inspect the subject's size in the training image.")
    if item.get("has_alpha") and not geometry["rgba"]:
        flags.append("Transparency is composited onto neutral grey for this model.")
    count = item.get("person_count") if kind == 'character' else None
    detection = item.get("detection", {}) if kind == 'character' else {}
    if count is None and detection.get("status") == "complete":
        count = detection.get("count")
        if detection.get("overlap_candidates"):
            flags.append(f"Overlapping person detections: {detection['count_min']}–{detection['count_max']} possible people. A head/body duplicate can cause this; review and override the count.")
    if count is not None and count > 1:
        flags.append("Multiple people reported. Review the target and caption; keep if intentional.")
    elif count == 0:
        flags.append("No person reported. A detector may miss close-ups, occluded or unusual poses.")
    if item.get("duplicate_candidates"):
        flags.append("Visually similar image candidate(s). Review for repeated frames; this is not identity matching.")
    return flags


def decorate(project, item):
    result = dict(item)
    if project['kind'] == 'character' and item.get("detection"):
        result["detection"] = assess_boxes(item["detection"])
    geometry = plan(project, item)
    result["plan"] = geometry
    result["preparation_stale"] = bool(item.get("caption")) and item.get("caption_preparation_digest") != geometry["digest"]
    result['effective_coverage'], result['effective_coverage_override'] = effective_coverage(project, result)
    flags = advisory(result, geometry, project['kind']) + category_advisories(project, result, geometry) + objects.issues(project, item)
    if project['kind'] == 'object':
        result['object_selection'] = objects.selection(project, item)
    evidence = [geometry['digest'], flags, result.get('detection'), item.get('person_count')] if project['kind'] == 'character' else [geometry['digest'], flags, project['kind'], result['effective_coverage']]
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    result.update(advisories=flags, advisory_digest=digest, advisories_accepted=item.get("advisory_override") == digest)
    return result
