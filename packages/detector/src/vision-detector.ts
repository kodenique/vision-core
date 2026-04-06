import type { EmbeddingEngine, ImageInput, DetectionModelConfig, DetectionResult } from '@vision-core/types';
import { preprocessImage } from '@vision-core/core';
import { decodeYoloOutput } from './yolo-decoder.js';

const YOLO_INPUT_SIZE = 640;

export class VisionDetector {
  private config: DetectionModelConfig | null = null;

  constructor(private readonly engine: EmbeddingEngine) {}

  async initialize(config: DetectionModelConfig): Promise<void> {
    this.config = config;
    await this.engine.loadModel(config);
  }

  async detect(image: ImageInput, originalWidth: number, originalHeight: number): Promise<DetectionResult> {
    if (!this.config) {
      throw new Error('VisionDetector not initialized. Call initialize() first.');
    }

    const tensor = preprocessImage(image, {
      inputWidth: YOLO_INPUT_SIZE,
      inputHeight: YOLO_INPUT_SIZE,
      normalization: { mean: [0, 0, 0], std: [1, 1, 1] },
      channelOrder: 'CHW',
    });

    const batchedTensor = { data: tensor.data, shape: [1, ...tensor.shape] };

    const result = await this.engine.extractEmbedding(batchedTensor);

    const rawDetections = decodeYoloOutput(
      result.embedding,
      this.config.confidenceThreshold,
      this.config.iouThreshold,
      this.config.maxDetections,
      this.config.classLabels
    );

    const xScale = originalWidth / YOLO_INPUT_SIZE;
    const yScale = originalHeight / YOLO_INPUT_SIZE;

    const detections = rawDetections.map(det => ({
      ...det,
      bbox: {
        x: det.bbox.x * xScale,
        y: det.bbox.y * yScale,
        width: det.bbox.width * xScale,
        height: det.bbox.height * yScale,
      },
    }));

    return { detections, modelId: result.modelId };
  }

  async dispose(): Promise<void> {
    await this.engine.dispose();
  }
}
