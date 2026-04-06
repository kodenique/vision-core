import { createVisionCore, EngineNotInitializedError } from '@vision-core/core';
import { createOnnxEngine } from '@vision-core/engine-onnx';
import type { OnnxRuntime, OnnxInferenceSession, OnnxTensor } from '@vision-core/engine-onnx';
import type { ImageInput, ModelConfig } from '@vision-core/types';
import goldenFixture from '../fixtures/golden-4x4-rgb.json';

// --- Mock OnnxRuntime ---

const MOCK_EMBEDDING_SIZE = 128;
const mockEmbeddingData = new Float32Array(MOCK_EMBEDDING_SIZE).fill(0.5);

function makeMockOnnxRuntime(outputTensorName = 'output'): OnnxRuntime {
  const session: OnnxInferenceSession = {
    run: jest.fn().mockImplementation(async () => {
      const result: Record<string, OnnxTensor> = {
        [outputTensorName]: { data: mockEmbeddingData, dims: [1, MOCK_EMBEDDING_SIZE] },
      };
      return result;
    }),
    release: jest.fn().mockResolvedValue(undefined),
  };
  return {
    InferenceSession: {
      create: jest.fn().mockResolvedValue(session),
    },
  };
}

// --- Model Config ---

const modelConfig: ModelConfig = {
  modelSource: 'mock://test-model',
  modelLoader: async () => new ArrayBuffer(16),
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 4,
  inputHeight: 4,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
};

// --- Helper: create a mock RGBA image ---

function createMockImage(width: number, height: number, fillValue = 128): ImageInput {
  return {
    data: new Uint8Array(width * height * 4).fill(fillValue),
    width,
    height,
  };
}

// --- Helper: inline normalization (matches normalizePixels CHW logic) ---

function normalizePixelsCHW(
  pixels: number[][],
  mean: [number, number, number],
  std: [number, number, number]
): Float32Array {
  const numPixels = pixels.length;
  const result = new Float32Array(numPixels * 3);
  for (let i = 0; i < numPixels; i++) {
    result[i]                = (pixels[i][0] / 255 - mean[0]) / std[0]; // R
    result[numPixels + i]    = (pixels[i][1] / 255 - mean[1]) / std[1]; // G
    result[numPixels * 2 + i] = (pixels[i][2] / 255 - mean[2]) / std[2]; // B
  }
  return result;
}

// -------------------------------------------------------------------

describe('Full pipeline integration', () => {
  describe('VisionCore with OnnxEmbeddingEngine (backend-only)', () => {
    it('initialize -> embed -> dispose returns correct EmbeddingResult', async () => {
      const runtime = makeMockOnnxRuntime();
      const engine = createOnnxEngine(runtime);
      const vc = createVisionCore(engine);

      await vc.initialize(modelConfig);

      const image = createMockImage(4, 4);
      const result = await vc.embed(image);

      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.dimensions).toBe(MOCK_EMBEDDING_SIZE);
      expect(result.modelId).toBe('mock://test-model');
      expect(result.embedding.length).toBe(MOCK_EMBEDDING_SIZE);

      await vc.dispose();
    });

    it('embed returns predictable embedding values from mock runtime', async () => {
      const runtime = makeMockOnnxRuntime();
      const engine = createOnnxEngine(runtime);
      const vc = createVisionCore(engine);

      await vc.initialize(modelConfig);
      const result = await vc.embed(createMockImage(4, 4));

      for (let i = 0; i < result.embedding.length; i++) {
        expect(result.embedding[i]).toBeCloseTo(0.5, 5);
      }

      await vc.dispose();
    });

    it('throws EngineNotInitializedError when embed called before initialize', async () => {
      const vc = createVisionCore(createOnnxEngine(makeMockOnnxRuntime()));
      await expect(vc.embed(createMockImage(4, 4))).rejects.toThrow(EngineNotInitializedError);
    });

    it('dispose resets state and prevents further embedding', async () => {
      const runtime = makeMockOnnxRuntime();
      const engine = createOnnxEngine(runtime);
      const vc = createVisionCore(engine);

      await vc.initialize(modelConfig);
      await vc.dispose();

      await expect(vc.embed(createMockImage(4, 4))).rejects.toThrow(EngineNotInitializedError);
    });
  });

  describe('Golden normalization fixture', () => {
    it('normalizes 4x4 uniform pixels to expected CHW values', () => {
      const { pixels, normalization, expectedOutput, epsilon } = goldenFixture;
      const mean = normalization.mean as [number, number, number];
      const std = normalization.std as [number, number, number];

      const computed = normalizePixelsCHW(pixels, mean, std);

      expect(computed.length).toBe(expectedOutput.length);
      for (let i = 0; i < computed.length; i++) {
        expect(Math.abs(computed[i] - expectedOutput[i])).toBeLessThanOrEqual(epsilon);
      }
    });

    it('golden file has correct shape: 3 channels x 16 pixels = 48 values', () => {
      expect(goldenFixture.pixels.length).toBe(16);
      expect(goldenFixture.expectedOutput.length).toBe(48);
    });
  });
});
