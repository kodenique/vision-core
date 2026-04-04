/**
 * Normalize RGBA Uint8Array pixel data into a Float32Array for model input.
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
