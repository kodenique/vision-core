import { resizePixelData } from '../resize.js';
import type { RawPixelData } from '../types.js';

function makeUniformRgba(width: number, height: number, r: number, g: number, b: number, a = 255): RawPixelData {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = a;
  }
  return { data, width, height };
}

describe('resizePixelData', () => {
  describe('bilinear resize 4x4 → 2x2', () => {
    it('samples exact source pixels when scale is integer and fractions are zero', () => {
      // 4x4 where the sampled pixels (0,0),(0,2),(2,0),(2,2) have distinct colors
      // With scale=2, dst(x,y) samples srcX=x*2, srcY=y*2 — fractions are 0
      const data = new Uint8Array(4 * 4 * 4);
      // fill with gray
      for (let i = 0; i < 64; i += 4) { data[i] = 128; data[i+1] = 128; data[i+2] = 128; data[i+3] = 255; }
      // pixel at (row=0, col=0): Red
      data[0] = 255; data[1] = 0; data[2] = 0; data[3] = 255;
      // pixel at (row=0, col=2): Green → index (0*4+2)*4 = 8
      data[8] = 0; data[9] = 255; data[10] = 0; data[11] = 255;
      // pixel at (row=2, col=0): Blue → index (2*4+0)*4 = 32
      data[32] = 0; data[33] = 0; data[34] = 255; data[35] = 255;
      // pixel at (row=2, col=2): Yellow → index (2*4+2)*4 = 40
      data[40] = 255; data[41] = 255; data[42] = 0; data[43] = 255;

      const src: RawPixelData = { data, width: 4, height: 4 };
      const result = resizePixelData(src, 2, 2);

      expect(result.width).toBe(2);
      expect(result.height).toBe(2);
      expect(result.data).toHaveLength(2 * 2 * 4);

      // dst(x=0,y=0) → srcX=0,srcY=0 → Red
      expect(result.data[0]).toBe(255);
      expect(result.data[1]).toBe(0);
      expect(result.data[2]).toBe(0);
      // dst(x=1,y=0) → srcX=2,srcY=0 → Green
      expect(result.data[4]).toBe(0);
      expect(result.data[5]).toBe(255);
      expect(result.data[6]).toBe(0);
      // dst(x=0,y=1) → srcX=0,srcY=2 → Blue
      expect(result.data[8]).toBe(0);
      expect(result.data[9]).toBe(0);
      expect(result.data[10]).toBe(255);
      // dst(x=1,y=1) → srcX=2,srcY=2 → Yellow
      expect(result.data[12]).toBe(255);
      expect(result.data[13]).toBe(255);
      expect(result.data[14]).toBe(0);
    });

    it('preserves uniform color across all pixels', () => {
      const src = makeUniformRgba(4, 4, 100, 150, 200);
      const result = resizePixelData(src, 2, 2);

      for (let i = 0; i < 4; i++) {
        expect(result.data[i * 4]).toBe(100);
        expect(result.data[i * 4 + 1]).toBe(150);
        expect(result.data[i * 4 + 2]).toBe(200);
      }
    });
  });

  describe('identity resize (same dimensions)', () => {
    it('returns equivalent pixel data when target matches source size', () => {
      const src = makeUniformRgba(3, 3, 42, 84, 126);
      const result = resizePixelData(src, 3, 3);

      expect(result.width).toBe(3);
      expect(result.height).toBe(3);
      expect(result.data).toHaveLength(3 * 3 * 4);
      for (let i = 0; i < 9; i++) {
        expect(result.data[i * 4]).toBe(42);
        expect(result.data[i * 4 + 1]).toBe(84);
        expect(result.data[i * 4 + 2]).toBe(126);
      }
    });
  });

  describe('non-square resize', () => {
    it('resizes 4x2 → 2x4 and reports correct dimensions', () => {
      const src = makeUniformRgba(4, 2, 60, 120, 180);
      const result = resizePixelData(src, 2, 4);

      expect(result.width).toBe(2);
      expect(result.height).toBe(4);
      expect(result.data).toHaveLength(2 * 4 * 4);
    });

    it('preserves uniform color through non-square resize', () => {
      const src = makeUniformRgba(4, 2, 60, 120, 180);
      const result = resizePixelData(src, 2, 4);

      for (let i = 0; i < 8; i++) {
        expect(result.data[i * 4]).toBe(60);
        expect(result.data[i * 4 + 1]).toBe(120);
        expect(result.data[i * 4 + 2]).toBe(180);
      }
    });

    it('resizes 1x4 → 4x1 and reports correct dimensions', () => {
      const src = makeUniformRgba(1, 4, 10, 20, 30);
      const result = resizePixelData(src, 4, 1);

      expect(result.width).toBe(4);
      expect(result.height).toBe(1);
      expect(result.data).toHaveLength(4 * 1 * 4);
    });
  });

  describe('output dimensions', () => {
    it('always returns the exact requested target dimensions', () => {
      const src = makeUniformRgba(8, 6, 0, 0, 0);
      const result = resizePixelData(src, 3, 5);

      expect(result.width).toBe(3);
      expect(result.height).toBe(5);
      expect(result.data).toHaveLength(3 * 5 * 4);
    });
  });
});
