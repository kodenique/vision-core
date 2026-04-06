import { VisionCore, createVisionCore, EngineNotInitializedError, InferenceError } from '../src/index';
import type { EmbeddingEngine, ImageInput, ModelConfig, EmbeddingResult } from '@vision-core/types';

const mockModelConfig: ModelConfig = {
  modelSource: 'test-model',
  modelLoader: async () => new ArrayBuffer(0),
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 4,
  inputHeight: 4,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
};

const mockEmbeddingResult: EmbeddingResult = {
  embedding: new Float32Array([0.1, 0.2, 0.3]),
  dimensions: 3,
  modelId: 'test-model',
};

// 4x4 RGBA image (64 bytes)
const mockImageInput: ImageInput = {
  data: new Uint8Array(4 * 4 * 4).fill(128),
  width: 4,
  height: 4,
};

function makeMockEngine(overrides?: Partial<EmbeddingEngine>): EmbeddingEngine {
  return {
    loadModel: jest.fn().mockResolvedValue(undefined),
    extractEmbedding: jest.fn().mockResolvedValue(mockEmbeddingResult),
    dispose: jest.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

describe('VisionCore', () => {
  describe('lifecycle', () => {
    it('initializes successfully', async () => {
      const engine = makeMockEngine();
      const vc = new VisionCore(engine);

      await vc.initialize(mockModelConfig);

      expect(engine.loadModel).toHaveBeenCalledWith(mockModelConfig);
    });

    it('embeds after initialization', async () => {
      const engine = makeMockEngine();
      const vc = new VisionCore(engine);

      await vc.initialize(mockModelConfig);
      const result = await vc.embed(mockImageInput);

      expect(engine.extractEmbedding).toHaveBeenCalled();
      expect(result).toEqual(mockEmbeddingResult);
    });

    it('disposes and resets initialized state', async () => {
      const engine = makeMockEngine();
      const vc = new VisionCore(engine);

      await vc.initialize(mockModelConfig);
      await vc.dispose();

      expect(engine.dispose).toHaveBeenCalled();
      await expect(vc.embed(mockImageInput)).rejects.toThrow(EngineNotInitializedError);
    });
  });

  describe('error handling', () => {
    it('throws EngineNotInitializedError when embed called before init', async () => {
      const vc = new VisionCore(makeMockEngine());

      await expect(vc.embed(mockImageInput)).rejects.toThrow(EngineNotInitializedError);
    });

    it('throws InferenceError when engine fails', async () => {
      const engine = makeMockEngine({
        extractEmbedding: jest.fn().mockRejectedValue(new Error('inference failed')),
      });
      const vc = new VisionCore(engine);
      await vc.initialize(mockModelConfig);

      await expect(vc.embed(mockImageInput)).rejects.toThrow(InferenceError);
    });
  });

  describe('happy path', () => {
    it('createVisionCore factory returns a VisionCore instance', () => {
      const vc = createVisionCore(makeMockEngine());
      expect(vc).toBeInstanceOf(VisionCore);
    });

    it('full flow: init -> embed -> dispose', async () => {
      const engine = makeMockEngine();
      const vc = createVisionCore(engine);

      await vc.initialize(mockModelConfig);
      const result = await vc.embed(mockImageInput);
      await vc.dispose();

      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.dimensions).toBe(3);
      expect(result.modelId).toBe('test-model');
    });
  });
});
