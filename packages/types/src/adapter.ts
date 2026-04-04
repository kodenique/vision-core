import type { ImageSize } from './model.js';
import type { TensorInput } from './tensor.js';

export interface ImageAdapter<TInput> {
  preprocess(input: TInput, targetSize: ImageSize): Promise<TensorInput>;
}
