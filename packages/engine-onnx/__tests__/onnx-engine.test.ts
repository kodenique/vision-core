import { OnnxEmbeddingEngine } from '../src/onnx-engine';
import { createOnnxEngine } from '../src/factory';
import type { OnnxRuntime, OnnxInferenceSession } from '../src/onnx-types';
import type { ModelConfig, TensorInput } from '@vision-core/types';

const makeSession = (overrides?: Partial<OnnxInferenceSession>): OnnxInferenceSession => ({
  run: jest.fn().mockResolvedValue({
    output: { data: new Float32Array([0.1, 0.2, 0.3]), dims: [1, 3] },
  }),
  release: jest.fn().mockResolvedValue(undefined),
  ...overrides,
});

const makeRuntime = (session: OnnxInferenceSession): OnnxRuntime => ({
  InferenceSession: {
    create: jest.fn().mockResolvedValue(session),
  },
});

const makeConfig = (overrides?: Partial<ModelConfig>): ModelConfig => ({
  modelSource: 'model.onnx',
  modelLoader: jest.fn().mockResolvedValue(new ArrayBuffer(8)),
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 224,
  inputHeight: 224,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
  ...overrides,
});

const makeTensorInput = (): TensorInput => ({
  data: new Float32Array([1, 2, 3, 4]),
  shape: [1, 3, 224, 224],
});

describe('OnnxEmbeddingEngine', () => {
  describe('loadModel', () => {
    it('calls modelLoader with modelSource and creates InferenceSession', async () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const config = makeConfig();
      const engine = new OnnxEmbeddingEngine(runtime);

      await engine.loadModel(config);

      expect(config.modelLoader).toHaveBeenCalledWith('model.onnx');
      expect(runtime.InferenceSession.create).toHaveBeenCalledWith(expect.any(ArrayBuffer));
    });

    it('throws if modelLoader rejects', async () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const config = makeConfig({
        modelLoader: jest.fn().mockRejectedValue(new Error('fetch failed')),
      });
      const engine = new OnnxEmbeddingEngine(runtime);

      await expect(engine.loadModel(config)).rejects.toThrow('fetch failed');
    });

    it('throws if InferenceSession.create rejects', async () => {
      const runtime: OnnxRuntime = {
        InferenceSession: {
          create: jest.fn().mockRejectedValue(new Error('invalid model')),
        },
      };
      const config = makeConfig();
      const engine = new OnnxEmbeddingEngine(runtime);

      await expect(engine.loadModel(config)).rejects.toThrow('invalid model');
    });
  });

  describe('extractEmbedding', () => {
    it('builds feeds using inputTensorName and reads output using outputTensorName', async () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const config = makeConfig({ inputTensorName: 'pixel_values', outputTensorName: 'embeddings' });
      const mockRun = jest.fn().mockResolvedValue({
        embeddings: { data: new Float32Array([0.5, 0.6]), dims: [1, 2] },
      });
      session.run = mockRun;

      const engine = new OnnxEmbeddingEngine(runtime);
      await engine.loadModel(config);

      const input = makeTensorInput();
      const result = await engine.extractEmbedding(input);

      expect(mockRun).toHaveBeenCalledWith({
        pixel_values: { data: input.data, dims: input.shape },
      });
      expect(result.embedding).toEqual(new Float32Array([0.5, 0.6]));
      expect(result.dimensions).toBe(2);
      expect(result.modelId).toBe('model.onnx');
    });

    it('converts number[] output to Float32Array', async () => {
      const session = makeSession({
        run: jest.fn().mockResolvedValue({
          output: { data: [1, 2, 3], dims: [1, 3] },
        }),
      });
      const runtime = makeRuntime(session);
      const engine = new OnnxEmbeddingEngine(runtime);
      await engine.loadModel(makeConfig());

      const result = await engine.extractEmbedding(makeTensorInput());

      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.embedding).toEqual(new Float32Array([1, 2, 3]));
    });

    it('throws if model not loaded', async () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const engine = new OnnxEmbeddingEngine(runtime);

      await expect(engine.extractEmbedding(makeTensorInput())).rejects.toThrow(
        'Model not loaded',
      );
    });

    it('throws if output tensor not found in results', async () => {
      const session = makeSession({
        run: jest.fn().mockResolvedValue({}),
      });
      const runtime = makeRuntime(session);
      const engine = new OnnxEmbeddingEngine(runtime);
      await engine.loadModel(makeConfig());

      await expect(engine.extractEmbedding(makeTensorInput())).rejects.toThrow(
        "Output tensor 'output' not found",
      );
    });

    it('throws if session.run rejects', async () => {
      const session = makeSession({
        run: jest.fn().mockRejectedValue(new Error('inference error')),
      });
      const runtime = makeRuntime(session);
      const engine = new OnnxEmbeddingEngine(runtime);
      await engine.loadModel(makeConfig());

      await expect(engine.extractEmbedding(makeTensorInput())).rejects.toThrow('inference error');
    });
  });

  describe('dispose', () => {
    it('calls session.release and clears state', async () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const engine = new OnnxEmbeddingEngine(runtime);
      await engine.loadModel(makeConfig());

      await engine.dispose();

      expect(session.release).toHaveBeenCalled();
      // After dispose, extractEmbedding should throw
      await expect(engine.extractEmbedding(makeTensorInput())).rejects.toThrow('Model not loaded');
    });

    it('is safe to call when no model is loaded', async () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const engine = new OnnxEmbeddingEngine(runtime);

      await expect(engine.dispose()).resolves.toBeUndefined();
      expect(session.release).not.toHaveBeenCalled();
    });
  });

  describe('createOnnxEngine', () => {
    it('returns an OnnxEmbeddingEngine instance', () => {
      const session = makeSession();
      const runtime = makeRuntime(session);
      const engine = createOnnxEngine(runtime);

      expect(engine).toBeInstanceOf(OnnxEmbeddingEngine);
    });
  });
});
