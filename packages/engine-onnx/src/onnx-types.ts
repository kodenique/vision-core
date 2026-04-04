export interface OnnxTensor {
  data: Float32Array | number[];
  dims: number[];
}

export interface OnnxInferenceSession {
  run(feeds: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>>;
  release(): Promise<void>;
}

export interface OnnxRuntime {
  InferenceSession: {
    create(data: ArrayBuffer, options?: unknown): Promise<OnnxInferenceSession>;
  };
}
