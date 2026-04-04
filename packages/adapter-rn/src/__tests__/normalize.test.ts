import { normalizePixelData } from '../normalize.js';

const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [number, number, number] = [0.229, 0.224, 0.225];
const EPSILON = 1e-6;

function makeRgbaPixel(r: number, g: number, b: number, a = 255): Uint8Array {
  return new Uint8Array([r, g, b, a]);
}

describe('normalizePixelData', () => {
  describe('CHW channel ordering', () => {
    it('outputs R channel first, then G, then B for a 1x1 image', () => {
      // R=200, G=100, B=50 → CHW: [R_norm, G_norm, B_norm]
      const data = makeRgbaPixel(200, 100, 50);
      const result = normalizePixelData(data, 1, 1, IMAGENET_MEAN, IMAGENET_STD, 'CHW');

      const rNorm = (200 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      const gNorm = (100 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      const bNorm = (50 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

      expect(result).toHaveLength(3);
      expect(result[0]).toBeCloseTo(rNorm, 6);
      expect(result[1]).toBeCloseTo(gNorm, 6);
      expect(result[2]).toBeCloseTo(bNorm, 6);
    });

    it('places all R values before all G values before all B values for multi-pixel image', () => {
      // 2x1 image: pixel0=[255,0,0,255], pixel1=[0,255,0,255]
      const data = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]);
      const result = normalizePixelData(data, 2, 1, IMAGENET_MEAN, IMAGENET_STD, 'CHW');

      // CHW: indices 0..1 = R channel, 2..3 = G channel, 4..5 = B channel
      expect(result).toHaveLength(6);
      // R channel (indices 0-1): pixel0_R and pixel1_R
      expect(result[0]).toBeCloseTo((255 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 6);
      expect(result[1]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 6);
      // G channel (indices 2-3)
      expect(result[2]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], 6);
      expect(result[3]).toBeCloseTo((255 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], 6);
      // B channel (indices 4-5): both 0
      expect(result[4]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], 6);
      expect(result[5]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], 6);
    });
  });

  describe('HWC channel ordering', () => {
    it('outputs interleaved RGB for a 1x1 image', () => {
      const data = makeRgbaPixel(200, 100, 50);
      const result = normalizePixelData(data, 1, 1, IMAGENET_MEAN, IMAGENET_STD, 'HWC');

      expect(result).toHaveLength(3);
      expect(result[0]).toBeCloseTo((200 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 6);
      expect(result[1]).toBeCloseTo((100 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], 6);
      expect(result[2]).toBeCloseTo((50 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], 6);
    });

    it('interleaves RGB per pixel for multi-pixel image', () => {
      // 2x1 image: pixel0=[255,0,0,255], pixel1=[0,255,0,255]
      const data = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]);
      const result = normalizePixelData(data, 2, 1, IMAGENET_MEAN, IMAGENET_STD, 'HWC');

      // HWC: [pixel0_R, pixel0_G, pixel0_B, pixel1_R, pixel1_G, pixel1_B]
      expect(result).toHaveLength(6);
      expect(result[0]).toBeCloseTo((255 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 6);
      expect(result[1]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], 6);
      expect(result[2]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], 6);
      expect(result[3]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 6);
      expect(result[4]).toBeCloseTo((255 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], 6);
      expect(result[5]).toBeCloseTo((0 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], 6);
    });
  });

  describe('ImageNet normalization math', () => {
    it('correctly normalizes pixel [128, 64, 192] with ImageNet mean/std', () => {
      // Expected: (pixel/255 - mean) / std
      // R: (128/255 - 0.485) / 0.229 ≈ 0.074065
      // G: (64/255  - 0.456) / 0.224 ≈ -0.915266
      // B: (192/255 - 0.406) / 0.225 ≈ 1.541961
      const data = makeRgbaPixel(128, 64, 192);
      const result = normalizePixelData(data, 1, 1, IMAGENET_MEAN, IMAGENET_STD, 'CHW');

      const expectedR = (128 / 255 - 0.485) / 0.229;
      const expectedG = (64 / 255 - 0.456) / 0.224;
      const expectedB = (192 / 255 - 0.406) / 0.225;

      expect(Math.abs(result[0] - expectedR)).toBeLessThan(EPSILON);
      expect(Math.abs(result[1] - expectedG)).toBeLessThan(EPSILON);
      expect(Math.abs(result[2] - expectedB)).toBeLessThan(EPSILON);
    });
  });

  describe('alpha channel', () => {
    it('skips the alpha channel in normalization', () => {
      // Two pixels with same RGB but different alpha — output should be identical
      const dataA = new Uint8Array([128, 64, 192, 0]);
      const dataB = new Uint8Array([128, 64, 192, 255]);

      const resultA = normalizePixelData(dataA, 1, 1, IMAGENET_MEAN, IMAGENET_STD, 'CHW');
      const resultB = normalizePixelData(dataB, 1, 1, IMAGENET_MEAN, IMAGENET_STD, 'CHW');

      expect(resultA[0]).toBeCloseTo(resultB[0], 6);
      expect(resultA[1]).toBeCloseTo(resultB[1], 6);
      expect(resultA[2]).toBeCloseTo(resultB[2], 6);
    });
  });

  describe('golden fixture: 4x4 known RGBA input', () => {
    it('produces exact normalized Float32Array within epsilon 1e-6 (CHW)', () => {
      // 4x4 image, all pixels = [128, 64, 192, 255]
      const numPixels = 16;
      const pixelData = new Uint8Array(numPixels * 4);
      for (let i = 0; i < numPixels; i++) {
        pixelData[i * 4] = 128;
        pixelData[i * 4 + 1] = 64;
        pixelData[i * 4 + 2] = 192;
        pixelData[i * 4 + 3] = 255;
      }

      const result = normalizePixelData(pixelData, 4, 4, IMAGENET_MEAN, IMAGENET_STD, 'CHW');

      const expectedR = (128 / 255 - 0.485) / 0.229;
      const expectedG = (64 / 255 - 0.456) / 0.224;
      const expectedB = (192 / 255 - 0.406) / 0.225;

      // CHW: first 16 = R plane, next 16 = G plane, last 16 = B plane
      for (let i = 0; i < 16; i++) {
        expect(Math.abs(result[i] - expectedR)).toBeLessThan(EPSILON);
      }
      for (let i = 16; i < 32; i++) {
        expect(Math.abs(result[i] - expectedG)).toBeLessThan(EPSILON);
      }
      for (let i = 32; i < 48; i++) {
        expect(Math.abs(result[i] - expectedB)).toBeLessThan(EPSILON);
      }
    });

    it('produces exact normalized Float32Array within epsilon 1e-6 (HWC)', () => {
      const numPixels = 16;
      const pixelData = new Uint8Array(numPixels * 4);
      for (let i = 0; i < numPixels; i++) {
        pixelData[i * 4] = 128;
        pixelData[i * 4 + 1] = 64;
        pixelData[i * 4 + 2] = 192;
        pixelData[i * 4 + 3] = 255;
      }

      const result = normalizePixelData(pixelData, 4, 4, IMAGENET_MEAN, IMAGENET_STD, 'HWC');

      const expectedR = (128 / 255 - 0.485) / 0.229;
      const expectedG = (64 / 255 - 0.456) / 0.224;
      const expectedB = (192 / 255 - 0.406) / 0.225;

      // HWC: [R,G,B] interleaved per pixel
      for (let i = 0; i < 16; i++) {
        expect(Math.abs(result[i * 3] - expectedR)).toBeLessThan(EPSILON);
        expect(Math.abs(result[i * 3 + 1] - expectedG)).toBeLessThan(EPSILON);
        expect(Math.abs(result[i * 3 + 2] - expectedB)).toBeLessThan(EPSILON);
      }
    });
  });
});
