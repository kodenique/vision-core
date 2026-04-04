import { RNImageAdapter } from '../rn-adapter.js';
import type { PixelDecoder, RawPixelData, RNImageInput } from '../types.js';
import type { ImageSize } from '@vision-core/types';

const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [number, number, number] = [0.229, 0.224, 0.225];
const EPSILON = 1e-6;

function makeUniformRgba(width: number, height: number, r: number, g: number, b: number): RawPixelData {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { data, width, height };
}

function makeMockDecoder(returnData: RawPixelData): PixelDecoder & { calls: Array<{ input: RNImageInput; w: number; h: number }> } {
  const calls: Array<{ input: RNImageInput; w: number; h: number }> = [];
  return {
    calls,
    decode(input: RNImageInput, targetWidth: number, targetHeight: number) {
      calls.push({ input, w: targetWidth, h: targetHeight });
      return Promise.resolve(returnData);
    },
  };
}

const TARGET: ImageSize = { width: 4, height: 4 };

describe('RNImageAdapter', () => {
  describe('preprocess() with CHW channel ordering', () => {
    it('returns TensorInput with CHW shape [3, H, W]', async () => {
      const pixelData = makeUniformRgba(4, 4, 128, 64, 192);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      const result = await adapter.preprocess({ uri: 'test.jpg' }, TARGET);

      expect(result.shape).toEqual([3, 4, 4]);
      expect(result.data).toBeInstanceOf(Float32Array);
      expect(result.data).toHaveLength(3 * 4 * 4);
    });
  });

  describe('preprocess() with HWC channel ordering', () => {
    it('returns TensorInput with HWC shape [H, W, 3]', async () => {
      const pixelData = makeUniformRgba(4, 4, 128, 64, 192);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'HWC',
      });

      const result = await adapter.preprocess({ uri: 'test.jpg' }, TARGET);

      expect(result.shape).toEqual([4, 4, 3]);
      expect(result.data).toBeInstanceOf(Float32Array);
      expect(result.data).toHaveLength(4 * 4 * 3);
    });
  });

  describe('decoder.decode() call contract', () => {
    it('calls decoder.decode() with the input and target dimensions', async () => {
      const pixelData = makeUniformRgba(4, 4, 100, 100, 100);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      const input: RNImageInput = { uri: 'photo.png' };
      await adapter.preprocess(input, { width: 4, height: 4 });

      expect(decoder.calls).toHaveLength(1);
      expect(decoder.calls[0].input).toBe(input);
      expect(decoder.calls[0].w).toBe(4);
      expect(decoder.calls[0].h).toBe(4);
    });

    it('passes RN asset number input to decoder unchanged', async () => {
      const pixelData = makeUniformRgba(4, 4, 100, 100, 100);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      await adapter.preprocess(42 as RNImageInput, TARGET);

      expect(decoder.calls[0].input).toBe(42);
    });
  });

  describe('resize fallback', () => {
    it('triggers bilinear resize when decoder returns wrong dimensions', async () => {
      // Decoder returns 8x8 but target is 4x4 — adapter should resize to 4x4
      const pixelData = makeUniformRgba(8, 8, 200, 100, 50);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      const result = await adapter.preprocess({ uri: 'img.jpg' }, TARGET);

      expect(result.shape).toEqual([3, 4, 4]);
      expect(result.data).toHaveLength(3 * 4 * 4);
    });

    it('does not resize when decoder returns correct dimensions', async () => {
      const pixelData = makeUniformRgba(4, 4, 200, 100, 50);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      const result = await adapter.preprocess({ uri: 'img.jpg' }, TARGET);

      // Only one call to decoder, no resize needed
      expect(decoder.calls).toHaveLength(1);
      expect(result.shape).toEqual([3, 4, 4]);
    });
  });

  describe('error propagation', () => {
    it('propagates error when decoder.decode() throws', async () => {
      const failingDecoder: PixelDecoder = {
        decode() {
          return Promise.reject(new Error('decode failed'));
        },
      };
      const adapter = new RNImageAdapter(failingDecoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      await expect(adapter.preprocess({ uri: 'bad.jpg' }, TARGET)).rejects.toThrow('decode failed');
    });
  });

  describe('golden fixture: 4x4 RGB with ImageNet normalization (CHW)', () => {
    it('normalized output matches expected values within epsilon 1e-6', async () => {
      // All pixels = [128, 64, 192, 255]
      const pixelData = makeUniformRgba(4, 4, 128, 64, 192);
      const decoder = makeMockDecoder(pixelData);
      const adapter = new RNImageAdapter(decoder, {
        normalization: { mean: IMAGENET_MEAN, std: IMAGENET_STD },
        channelOrder: 'CHW',
      });

      const result = await adapter.preprocess({ uri: 'fixture.jpg' }, TARGET);

      const expectedR = (128 / 255 - 0.485) / 0.229;
      const expectedG = (64 / 255 - 0.456) / 0.224;
      const expectedB = (192 / 255 - 0.406) / 0.225;

      // CHW: first 16 = R plane, next 16 = G plane, last 16 = B plane
      for (let i = 0; i < 16; i++) {
        expect(Math.abs(result.data[i] - expectedR)).toBeLessThan(EPSILON);
      }
      for (let i = 16; i < 32; i++) {
        expect(Math.abs(result.data[i] - expectedG)).toBeLessThan(EPSILON);
      }
      for (let i = 32; i < 48; i++) {
        expect(Math.abs(result.data[i] - expectedB)).toBeLessThan(EPSILON);
      }
    });
  });
});
