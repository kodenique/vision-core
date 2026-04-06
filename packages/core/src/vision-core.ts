import type {
  EmbeddingEngine,
  ImageInput,
  ModelConfig,
  EmbeddingResult,
} from '@vision-core/types';
import {
  EngineNotInitializedError,
  InferenceError,
} from './errors.js';
import { preprocessImage } from './preprocessing.js';

export class VisionCore {
  private initialized = false;
  private config: ModelConfig | null = null;

  constructor(
    private readonly engine: EmbeddingEngine
  ) {}

  async initialize(config: ModelConfig): Promise<void> {
    await this.engine.loadModel(config);
    this.config = config;
    this.initialized = true;
  }

  async embed(image: ImageInput): Promise<EmbeddingResult> {
    if (!this.initialized) {
      throw new EngineNotInitializedError();
    }

    const tensorInput = preprocessImage(image, this.config!);

    let result;
    try {
      result = await this.engine.extractEmbedding(tensorInput);
    } catch (err) {
      throw new InferenceError(
        `Engine inference failed: ${err instanceof Error ? err.message : String(err)}`,
        err
      );
    }

    return result;
  }

  async dispose(): Promise<void> {
    await this.engine.dispose();
    this.initialized = false;
  }
}
