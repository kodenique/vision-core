import type { ImageAdapter, ImageSize, ModelConfig, TensorInput } from '@vision-core/types';
import type { PixelDecoder, RNImageInput } from './types.js';
import { resizePixelData } from './resize.js';
import { normalizePixelData } from './normalize.js';

export class RNImageAdapter implements ImageAdapter<RNImageInput> {
  private readonly decoder: PixelDecoder;
  private readonly normalization: ModelConfig['normalization'];
  private readonly channelOrder: ModelConfig['channelOrder'];

  constructor(
    decoder: PixelDecoder,
    config: Pick<ModelConfig, 'normalization' | 'channelOrder'>
  ) {
    this.decoder = decoder;
    this.normalization = config.normalization;
    this.channelOrder = config.channelOrder;
  }

  async preprocess(input: RNImageInput, targetSize: ImageSize): Promise<TensorInput> {
    const { width, height } = targetSize;

    let pixelData = await this.decoder.decode(input, width, height);

    // Fall back to built-in bilinear resize if decoder returned wrong dimensions
    if (pixelData.width !== width || pixelData.height !== height) {
      pixelData = resizePixelData(pixelData, width, height);
    }

    const data = normalizePixelData(
      pixelData.data,
      width,
      height,
      this.normalization.mean,
      this.normalization.std,
      this.channelOrder
    );

    const shape =
      this.channelOrder === 'CHW' ? [3, height, width] : [height, width, 3];

    return { data, shape };
  }
}
