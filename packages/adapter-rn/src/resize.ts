import type { RawPixelData } from './types.js';

/**
 * Bilinear interpolation resize for RGBA Uint8Array pixel data.
 */
export function resizePixelData(
  src: RawPixelData,
  targetWidth: number,
  targetHeight: number
): RawPixelData {
  const { data, width: srcWidth, height: srcHeight } = src;
  const output = new Uint8Array(targetWidth * targetHeight * 4);

  const xScale = srcWidth / targetWidth;
  const yScale = srcHeight / targetHeight;

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = x * xScale;
      const srcY = y * yScale;

      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, srcWidth - 1);
      const y1 = Math.min(y0 + 1, srcHeight - 1);

      const xFrac = srcX - x0;
      const yFrac = srcY - y0;

      const dstIdx = (y * targetWidth + x) * 4;

      for (let c = 0; c < 4; c++) {
        const tl = data[(y0 * srcWidth + x0) * 4 + c];
        const tr = data[(y0 * srcWidth + x1) * 4 + c];
        const bl = data[(y1 * srcWidth + x0) * 4 + c];
        const br = data[(y1 * srcWidth + x1) * 4 + c];

        const top = tl + (tr - tl) * xFrac;
        const bottom = bl + (br - bl) * xFrac;
        output[dstIdx + c] = Math.round(top + (bottom - top) * yFrac);
      }
    }
  }

  return { data: output, width: targetWidth, height: targetHeight };
}
