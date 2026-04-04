import type {
  EmbeddingEngine,
  ImageAdapter,
  ModelConfig,
  EmbeddingResult,
} from '@vision-core/types';
import {
  EngineNotInitializedError,
  InferenceError,
  AdapterError,
} from './errors.js';

export class VisionCore<TInput = unknown> {
  private initialized = false;

  constructor(
    private readonly engine: EmbeddingEngine,
    private readonly adapter: ImageAdapter<TInput>
  ) {}

  async initialize(config: ModelConfig): Promise<void> {
    await this.engine.loadModel(config);
    this.initialized = true;
  }

  async embed(imageInput: TInput): Promise<EmbeddingResult> {
    if (!this.initialized) {
      throw new EngineNotInitializedError();
    }

    const targetSize = { width: 0, height: 0 }; // resolved via config in real use
    let tensorInput;
    try {
      tensorInput = await this.adapter.preprocess(imageInput, targetSize);
    } catch (err) {
      throw new AdapterError(
        `Adapter preprocessing failed: ${err instanceof Error ? err.message : String(err)}`,
        err
      );
    }

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
