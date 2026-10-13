const clamp = (v, low, high) => Math.max(low, Math.min(high, v));

// All rectangles use the EXIF-oriented original's normalized coordinates.
export function moveCrop(rect, dx, dy) {
  dx = clamp(dx, -rect[0], 1 - rect[2]);
  dy = clamp(dy, -rect[1], 1 - rect[3]);
  return [rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy];
}

export function fitCropRatio(rect, ratio) {
  if (!ratio) return [...rect];
  const cx = (rect[0] + rect[2]) / 2, cy = (rect[1] + rect[3]) / 2;
  let w = rect[2] - rect[0], h = rect[3] - rect[1];
  if (w / h > ratio) w = h * ratio; else h = w / ratio;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

export function resizeCrop(rect, handle, dx, dy, ratio = 0, minX = .001, minY = .001) {
  let [l, t, r, b] = rect;
  const west = handle.includes('w'), east = handle.includes('e');
  const north = handle.includes('n'), south = handle.includes('s');
  if (!ratio) {
    if (west) l = clamp(l + dx, 0, r - minX);
    if (east) r = clamp(r + dx, l + minX, 1);
    if (north) t = clamp(t + dy, 0, b - minY);
    if (south) b = clamp(b + dy, t + minY, 1);
    return [l, t, r, b];
  }
  const ax = west ? r : east ? l : (l + r) / 2;
  const ay = north ? b : south ? t : (t + b) / 2;
  let w = r - l + (west ? -dx : east ? dx : 0);
  let h = b - t + (north ? -dy : south ? dy : 0);
  if (!(west || east)) w = h * ratio;
  else if (!(north || south) || Math.abs(dx) >= Math.abs(dy * ratio)) h = w / ratio;
  else w = h * ratio;
  const maxW = west ? ax : east ? 1 - ax : 2 * Math.min(ax, 1 - ax);
  const maxH = north ? ay : south ? 1 - ay : 2 * Math.min(ay, 1 - ay);
  w = Math.min(Math.max(w, minX, minY * ratio), maxW, maxH * ratio);
  h = w / ratio;
  l = west ? ax - w : east ? ax : ax - w / 2;
  t = north ? ay - h : south ? ay : ay - h / 2;
  return [l, t, l + w, t + h];
}

export function cropFromPlan(plan) {
  if (!plan) return [0, 0, 1, 1];
  const [l, t, r, b] = plan.source_region || [0, 0, plan.source_width, plan.source_height];
  return [l / plan.source_width, t / plan.source_height, r / plan.source_width, b / plan.source_height];
}
