import type { ImageAdapter, ImageSize, ModelConfig, TensorInput } from '@vision-core/types';
import {
  loadImageFromBlob,
  loadImageFromUrl,
  resizeImage,
  normalizePixels,
} from './image-utils';

export type WebImageInput =
  | File
  | Blob
  | HTMLImageElement
  | ImageBitmap
  | string;

export class WebImageAdapter implements ImageAdapter<WebImageInput> {
  private readonly normalization: ModelConfig['normalization'];
  private readonly channelOrder: ModelConfig['channelOrder'];

  constructor(config: Pick<ModelConfig, 'normalization' | 'channelOrder'>) {
    this.normalization = config.normalization;
    this.channelOrder = config.channelOrder;
  }

  async preprocess(input: WebImageInput, targetSize: ImageSize): Promise<TensorInput> {
    const { width, height } = targetSize;

    let bitmap: ImageBitmap;

    if (typeof input === 'string') {
      bitmap = await loadImageFromUrl(input);
    } else if (input instanceof HTMLImageElement) {
      bitmap = await createImageBitmap(input);
    } else if (input instanceof Blob) {
      // File extends Blob — covers both File and Blob inputs
      bitmap = await loadImageFromBlob(input);
    } else {
      bitmap = input;
    }

    const imageData = resizeImage(bitmap, width, height);
    const data = normalizePixels(
      imageData,
      this.normalization.mean,
      this.normalization.std,
      this.channelOrder
    );

    const shape =
      this.channelOrder === 'CHW' ? [3, height, width] : [height, width, 3];

    return { data, shape };
  }
}
