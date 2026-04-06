import type { BoundingBox } from '@vision-core/types';

export function computeIoU(a: BoundingBox, b: BoundingBox): number {
  const xA = Math.max(a.x, b.x);
  const yA = Math.max(a.y, b.y);
  const xB = Math.min(a.x + a.width, b.x + b.width);
  const yB = Math.min(a.y + a.height, b.y + b.height);

  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;

  if (interArea === 0) return 0;

  const aArea = a.width * a.height;
  const bArea = b.width * b.height;

  return interArea / (aArea + bArea - interArea);
}

export function nms(boxes: BoundingBox[], scores: number[], iouThreshold: number): number[] {
  const indices = scores
    .map((score, i) => ({ score, i }))
    .sort((a, b) => b.score - a.score)
    .map(({ i }) => i);

  const keep: number[] = [];
  const suppressed = new Set<number>();

  for (const i of indices) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of indices) {
      if (j === i || suppressed.has(j)) continue;
      if (computeIoU(boxes[i], boxes[j]) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return keep;
}
