import type { ModelConfig } from './model.js';
import type { TensorInput } from './tensor.js';

export type BoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Detection = {
  bbox: BoundingBox;
  label: string;
  classId: number;
  confidence: number;
};

export type DetectionResult = {
  detections: Detection[];
  modelId: string;
};

export type DetectionModelConfig = ModelConfig & {
  classLabels: string[];
  confidenceThreshold: number;
  iouThreshold: number;
  maxDetections: number;
};

export type DetectionEngine = {
  loadModel(config: DetectionModelConfig): Promise<void>;
  detect(input: TensorInput): Promise<DetectionResult>;
  dispose(): Promise<void>;
};
