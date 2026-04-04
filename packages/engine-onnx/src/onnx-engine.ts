import type { EmbeddingEngine, ModelConfig, TensorInput, EmbeddingResult } from '@vision-core/types';
import type { OnnxRuntime, OnnxInferenceSession } from './onnx-types.js';

export class OnnxEmbeddingEngine implements EmbeddingEngine {
  private session: OnnxInferenceSession | null = null;
  private config: ModelConfig | null = null;

  constructor(private readonly runtime: OnnxRuntime) {}

  async loadModel(config: ModelConfig): Promise<void> {
    this.config = config;
    const arrayBuffer = await config.modelLoader(config.modelSource);
    this.session = await this.runtime.InferenceSession.create(arrayBuffer);
  }

  async extractEmbedding(input: TensorInput): Promise<EmbeddingResult> {
    if (!this.session || !this.config) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    const feeds: Record<string, { data: Float32Array | number[]; dims: number[] }> = {
      [this.config.inputTensorName]: { data: input.data, dims: input.shape },
    };

    const results = await this.session.run(feeds);
    const outputTensor = results[this.config.outputTensorName];

    if (!outputTensor) {
      throw new Error(`Output tensor '${this.config.outputTensorName}' not found in results.`);
    }

    const embedding =
      outputTensor.data instanceof Float32Array
        ? outputTensor.data
        : new Float32Array(outputTensor.data);

    return {
      embedding,
      dimensions: embedding.length,
      modelId: this.config.modelSource,
    };
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.config = null;
  }
}
