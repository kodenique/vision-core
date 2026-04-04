export { VisionCore } from './vision-core.js';
export {
  EngineNotInitializedError,
  InvalidInputError,
  InferenceError,
  AdapterError,
} from './errors.js';
export { validateTensorInput } from './validation.js';

import type { EmbeddingEngine, ImageAdapter } from '@vision-core/types';
import { VisionCore } from './vision-core.js';

export function createVisionCore<TInput>(
  engine: EmbeddingEngine,
  adapter: ImageAdapter<TInput>
): VisionCore<TInput> {
  return new VisionCore(engine, adapter);
}
