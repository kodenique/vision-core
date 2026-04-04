import { VisionCore, createVisionCore, EngineNotInitializedError, InferenceError, AdapterError } from '../src/index';
import type { EmbeddingEngine, ImageAdapter, ModelConfig, TensorInput, EmbeddingResult } from '@vision-core/types';

const mockModelConfig: ModelConfig = {
  modelSource: 'test-model',
  modelLoader: async () => new ArrayBuffer(0),
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 224,
  inputHeight: 224,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
};

const mockTensorInput: TensorInput = {
  data: new Float32Array(224 * 224 * 3),
  shape: [1, 3, 224, 224],
};

const mockEmbeddingResult: EmbeddingResult = {
  embedding: new Float32Array([0.1, 0.2, 0.3]),
  dimensions: 3,
  modelId: 'test-model',
};

function makeMockEngine(overrides?: Partial<EmbeddingEngine>): EmbeddingEngine {
  return {
    loadModel: jest.fn().mockResolvedValue(undefined),
    extractEmbedding: jest.fn().mockResolvedValue(mockEmbeddingResult),
    dispose: jest.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

function makeMockAdapter(overrides?: Partial<ImageAdapter<unknown>>): ImageAdapter<unknown> {
  return {
    preprocess: jest.fn().mockResolvedValue(mockTensorInput),
    ...overrides,
  };
}

describe('VisionCore', () => {
  describe('lifecycle', () => {
    it('initializes successfully', async () => {
      const engine = makeMockEngine();
      const adapter = makeMockAdapter();
      const vc = new VisionCore(engine, adapter);

      await vc.initialize(mockModelConfig);

      expect(engine.loadModel).toHaveBeenCalledWith(mockModelConfig);
    });

    it('embeds after initialization', async () => {
      const engine = makeMockEngine();
      const adapter = makeMockAdapter();
      const vc = new VisionCore(engine, adapter);

      await vc.initialize(mockModelConfig);
      const result = await vc.embed('test-image');

      expect(adapter.preprocess).toHaveBeenCalled();
      expect(engine.extractEmbedding).toHaveBeenCalledWith(mockTensorInput);
      expect(result).toEqual(mockEmbeddingResult);
    });

    it('disposes and resets initialized state', async () => {
      const engine = makeMockEngine();
      const adapter = makeMockAdapter();
      const vc = new VisionCore(engine, adapter);

      await vc.initialize(mockModelConfig);
      await vc.dispose();

      expect(engine.dispose).toHaveBeenCalled();
      await expect(vc.embed('test-image')).rejects.toThrow(EngineNotInitializedError);
    });
  });

  describe('error handling', () => {
    it('throws EngineNotInitializedError when embed called before init', async () => {
      const vc = new VisionCore(makeMockEngine(), makeMockAdapter());

      await expect(vc.embed('test-image')).rejects.toThrow(EngineNotInitializedError);
    });

    it('throws AdapterError when adapter fails', async () => {
      const engine = makeMockEngine();
      const adapter = makeMockAdapter({
        preprocess: jest.fn().mockRejectedValue(new Error('preprocess failed')),
      });
      const vc = new VisionCore(engine, adapter);
      await vc.initialize(mockModelConfig);

      await expect(vc.embed('bad-input')).rejects.toThrow(AdapterError);
    });

    it('throws InferenceError when engine fails', async () => {
      const engine = makeMockEngine({
        extractEmbedding: jest.fn().mockRejectedValue(new Error('inference failed')),
      });
      const adapter = makeMockAdapter();
      const vc = new VisionCore(engine, adapter);
      await vc.initialize(mockModelConfig);

      await expect(vc.embed('test-image')).rejects.toThrow(InferenceError);
    });
  });

  describe('happy path', () => {
    it('createVisionCore factory returns a VisionCore instance', () => {
      const vc = createVisionCore(makeMockEngine(), makeMockAdapter());
      expect(vc).toBeInstanceOf(VisionCore);
    });

    it('full flow: init -> embed -> dispose', async () => {
      const engine = makeMockEngine();
      const adapter = makeMockAdapter();
      const vc = createVisionCore(engine, adapter);

      await vc.initialize(mockModelConfig);
      const result = await vc.embed('image-data');
      await vc.dispose();

      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.dimensions).toBe(3);
      expect(result.modelId).toBe('test-model');
    });
  });
});
