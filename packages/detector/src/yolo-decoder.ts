import type { Detection } from '@vision-core/types';
import { nms } from './nms.js';

const NUM_CLASSES = 80;
const NUM_PREDICTIONS = 8400;

export function decodeYoloOutput(
  raw: Float32Array,
  confidenceThreshold: number,
  iouThreshold: number,
  maxDetections: number,
  classLabels: string[]
): Detection[] {
  // raw is flattened [1, 84, 8400] — we access as [84, 8400] (batch dim ignored)
  // Layout: raw[(row * NUM_PREDICTIONS) + col] where row ∈ [0..83], col ∈ [0..8399]
  type Candidate = { bbox: { x: number; y: number; width: number; height: number }; classId: number; confidence: number };

  const candidates: Candidate[] = [];

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const cx = raw[0 * NUM_PREDICTIONS + i];
    const cy = raw[1 * NUM_PREDICTIONS + i];
    const w  = raw[2 * NUM_PREDICTIONS + i];
    const h  = raw[3 * NUM_PREDICTIONS + i];

    let bestScore = 0;
    let bestClass = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = raw[(4 + c) * NUM_PREDICTIONS + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < confidenceThreshold) continue;

    candidates.push({
      bbox: { x: cx - w / 2, y: cy - h / 2, width: w, height: h },
      classId: bestClass,
      confidence: bestScore,
    });
  }

  // Apply NMS per class
  const classGroups = new Map<number, Candidate[]>();
  for (const cand of candidates) {
    if (!classGroups.has(cand.classId)) classGroups.set(cand.classId, []);
    classGroups.get(cand.classId)!.push(cand);
  }

  const detections: Detection[] = [];
  for (const [classId, preds] of classGroups) {
    const boxes = preds.map(p => p.bbox);
    const scores = preds.map(p => p.confidence);
    const kept = nms(boxes, scores, iouThreshold);
    for (const idx of kept) {
      detections.push({
        bbox: preds[idx].bbox,
        label: classLabels[classId] ?? String(classId),
        classId,
        confidence: preds[idx].confidence,
      });
    }
  }

  detections.sort((a, b) => b.confidence - a.confidence);
  return detections.slice(0, maxDetections);
}
