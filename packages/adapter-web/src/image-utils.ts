export async function loadImageFromBlob(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob);
}

export async function loadImageFromUrl(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  const blob = await response.blob();
  return createImageBitmap(blob);
}

export function resizeImage(
  image: ImageBitmap,
  width: number,
  height: number
): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get 2D canvas context');
  ctx.drawImage(image, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

export function normalizePixels(
  imageData: ImageData,
  mean: [number, number, number],
  std: [number, number, number],
  channelOrder: 'CHW' | 'HWC'
): Float32Array {
  const { data, width, height } = imageData;
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
