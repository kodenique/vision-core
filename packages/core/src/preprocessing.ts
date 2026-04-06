import type { ImageInput, TensorInput, ModelConfig } from '@vision-core/types';

/**
 * Bilinear interpolation resize for RGBA Uint8Array pixel data.
 */
export function resizePixelData(
  src: ImageInput,
  targetWidth: number,
  targetHeight: number
): ImageInput {
  if (src.width === targetWidth && src.height === targetHeight) {
    return src;
  }

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

/**
 * Normalize RGBA pixel data into a Float32Array tensor.
 * Extracts RGB channels (skips alpha), normalizes to [0,1], applies mean/std.
 */
export function normalizePixelData(
  data: Uint8Array,
  width: number,
  height: number,
  mean: [number, number, number],
  std: [number, number, number],
  channelOrder: 'CHW' | 'HWC'
): Float32Array {
  const numPixels = width * height;
  const result = new Float32Array(numPixels * 3);

  if (channelOrder === 'CHW') {
    for (let i = 0; i < numPixels; i++) {
      result[i] = (data[i * 4] / 255 - mean[0]) / std[0];
      result[numPixels + i] = (data[i * 4 + 1] / 255 - mean[1]) / std[1];
      result[numPixels * 2 + i] = (data[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
  } else {
    for (let i = 0; i < numPixels; i++) {
      result[i * 3] = (data[i * 4] / 255 - mean[0]) / std[0];
      result[i * 3 + 1] = (data[i * 4 + 1] / 255 - mean[1]) / std[1];
      result[i * 3 + 2] = (data[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
  }

  return result;
}

/**
 * Preprocess raw RGBA pixel data into a model-ready tensor.
 * Resizes to target dimensions and normalizes per model config.
 */
export function preprocessImage(
  image: ImageInput,
  config: Pick<ModelConfig, 'inputWidth' | 'inputHeight' | 'normalization' | 'channelOrder'>
): TensorInput {
  const { inputWidth: width, inputHeight: height } = config;

  const resized = resizePixelData(image, width, height);

  const data = normalizePixelData(
    resized.data,
    width,
    height,
    config.normalization.mean,
    config.normalization.std,
    config.channelOrder
  );

  const shape =
    config.channelOrder === 'CHW' ? [3, height, width] : [height, width, 3];

  return { data, shape };
}
