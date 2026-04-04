import type { ModelConfig } from '@vision-core/types';
import type { PixelDecoder } from './types.js';
import { RNImageAdapter } from './rn-adapter.js';

export function createRNAdapter(
  decoder: PixelDecoder,
  config: Pick<ModelConfig, 'normalization' | 'channelOrder'>
): RNImageAdapter {
  return new RNImageAdapter(decoder, config);
}
