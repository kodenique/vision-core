import type { OnnxRuntime } from './onnx-types.js';
import { OnnxEmbeddingEngine } from './onnx-engine.js';

export function createOnnxEngine(onnxRuntime: OnnxRuntime): OnnxEmbeddingEngine {
  return new OnnxEmbeddingEngine(onnxRuntime);
}
