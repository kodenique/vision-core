import type { ModelConfig } from './model.js';
import type { TensorInput } from './tensor.js';
import type { EmbeddingResult } from './result.js';

export interface EmbeddingEngine {
  loadModel(config: ModelConfig): Promise<void>;
  extractEmbedding(input: TensorInput): Promise<EmbeddingResult>;
  dispose(): Promise<void>;
}
