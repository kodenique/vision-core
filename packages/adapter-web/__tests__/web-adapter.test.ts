import 'jest-canvas-mock';
import { normalizePixels, resizeImage } from '../src/image-utils';
import { WebImageAdapter } from '../src/web-adapter';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeImageData(
  width: number,
  height: number,
  rgb: [number, number, number]
): ImageData {
  const numPixels = width * height;
  const raw = new Uint8ClampedArray(numPixels * 4);
  for (let i = 0; i < numPixels; i++) {
    raw[i * 4] = rgb[0];
    raw[i * 4 + 1] = rgb[1];
    raw[i * 4 + 2] = rgb[2];
    raw[i * 4 + 3] = 255;
  }
  return new ImageData(raw, width, height);
}

// ImageNet defaults
const MEAN: [number, number, number] = [0.485, 0.456, 0.406];
const STD: [number, number, number] = [0.229, 0.224, 0.225];

// ---------------------------------------------------------------------------
// normalizePixels — unit tests (pure function, no canvas needed)
// ---------------------------------------------------------------------------

describe('normalizePixels', () => {
  // GOLDEN-FILE TEST
  // Pre-computed expected values for uniform pixel [128, 64, 32]:
  //   R = (128/255 - 0.485) / 0.229
  //   G = (64/255  - 0.456) / 0.224
  //   B = (32/255  - 0.406) / 0.225
  it('golden-file: 4×4 image all pixels [128,64,32], CHW layout, ImageNet normalization', () => {
    const width = 4;
    const height = 4;
    const numPixels = width * height;
    const imageData = makeImageData(width, height, [128, 64, 32]);

    const result = normalizePixels(imageData, MEAN, STD, 'CHW');

    expect(result).toHaveLength(numPixels * 3);

    const expectedR = (128 / 255 - 0.485) / 0.229;
    const expectedG = (64 / 255 - 0.456) / 0.224;
    const expectedB = (32 / 255 - 0.406) / 0.225;
    const epsilon = 1e-6;

    // Channel 0 — R (indices 0..15)
    for (let i = 0; i < numPixels; i++) {
      expect(Math.abs(result[i] - expectedR)).toBeLessThan(epsilon);
    }
    // Channel 1 — G (indices 16..31)
    for (let i = 0; i < numPixels; i++) {
      expect(Math.abs(result[numPixels + i] - expectedG)).toBeLessThan(epsilon);
    }
    // Channel 2 — B (indices 32..47)
    for (let i = 0; i < numPixels; i++) {
      expect(Math.abs(result[numPixels * 2 + i] - expectedB)).toBeLessThan(epsilon);
    }
  });

  it('golden-file: 4×4 image all pixels [128,64,32], HWC layout, ImageNet normalization', () => {
    const width = 4;
    const height = 4;
    const numPixels = width * height;
    const imageData = makeImageData(width, height, [128, 64, 32]);

    const result = normalizePixels(imageData, MEAN, STD, 'HWC');

    expect(result).toHaveLength(numPixels * 3);

    const expectedR = (128 / 255 - 0.485) / 0.229;
    const expectedG = (64 / 255 - 0.456) / 0.224;
    const expectedB = (32 / 255 - 0.406) / 0.225;
    const epsilon = 1e-6;

    for (let i = 0; i < numPixels; i++) {
      expect(Math.abs(result[i * 3] - expectedR)).toBeLessThan(epsilon);
      expect(Math.abs(result[i * 3 + 1] - expectedG)).toBeLessThan(epsilon);
      expect(Math.abs(result[i * 3 + 2] - expectedB)).toBeLessThan(epsilon);
    }
  });

  it('output length is width * height * 3', () => {
    const imageData = makeImageData(8, 6, [100, 150, 200]);
    const result = normalizePixels(imageData, MEAN, STD, 'CHW');
    expect(result).toHaveLength(8 * 6 * 3);
  });
});

// ---------------------------------------------------------------------------
// resizeImage — uses a canvas element as source (accepted by jest-canvas-mock)
// ---------------------------------------------------------------------------

describe('resizeImage', () => {
  it('returns ImageData with correct dimensions', () => {
    // Use an HTMLCanvasElement as source — jest-canvas-mock accepts it in drawImage
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = 64;
    sourceCanvas.height = 64;
    const imageData = resizeImage(sourceCanvas as unknown as ImageBitmap, 32, 32);
    expect(imageData.width).toBe(32);
    expect(imageData.height).toBe(32);
    expect(imageData.data).toHaveLength(32 * 32 * 4);
  });
});

// ---------------------------------------------------------------------------
// WebImageAdapter — integration tests with resizeImage mocked
// ---------------------------------------------------------------------------

// Mock resizeImage to return zeroed ImageData without needing actual canvas drawing
jest.mock('../src/image-utils', () => {
  const actual = jest.requireActual('../src/image-utils') as typeof import('../src/image-utils');
  return {
    ...actual,
    resizeImage: (_image: unknown, width: number, height: number): ImageData => {
      const data = new Uint8ClampedArray(width * height * 4);
      return new ImageData(data, width, height);
    },
  };
});

describe('WebImageAdapter', () => {
  const config = {
    normalization: { mean: MEAN, std: STD },
    channelOrder: 'CHW' as const,
  };
  const targetSize = { width: 4, height: 4 };

  it('returns TensorInput with correct shape for CHW', async () => {
    const adapter = new WebImageAdapter(config);
    const bitmap = { width: 4, height: 4 } as unknown as ImageBitmap;
    const result = await adapter.preprocess(bitmap, targetSize);

    expect(result.shape).toEqual([3, 4, 4]);
    expect(result.data).toBeInstanceOf(Float32Array);
    expect(result.data).toHaveLength(3 * 4 * 4);
  });

  it('returns TensorInput with correct shape for HWC', async () => {
    const adapter = new WebImageAdapter({
      normalization: { mean: MEAN, std: STD },
      channelOrder: 'HWC',
    });
    const bitmap = { width: 4, height: 4 } as unknown as ImageBitmap;
    const result = await adapter.preprocess(bitmap, targetSize);

    expect(result.shape).toEqual([4, 4, 3]);
    expect(result.data).toBeInstanceOf(Float32Array);
    expect(result.data).toHaveLength(4 * 4 * 3);
  });

  it('accepts a Blob input', async () => {
    const adapter = new WebImageAdapter(config);
    const blob = new Blob([''], { type: 'image/png' });
    // createImageBitmap is mocked by jest-canvas-mock
    const result = await adapter.preprocess(blob, targetSize);
    expect(result.data).toBeInstanceOf(Float32Array);
  });
});
