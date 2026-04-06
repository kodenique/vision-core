export { VisionCore } from './vision-core.js';
export {
  EngineNotInitializedError,
  InvalidInputError,
  InferenceError,
} from './errors.js';
export { validateTensorInput } from './validation.js';
export { preprocessImage, resizePixelData, normalizePixelData } from './preprocessing.js';

import type { EmbeddingEngine } from '@vision-core/types';
import { VisionCore } from './vision-core.js';

export function createVisionCore(engine: EmbeddingEngine): VisionCore {
  return new VisionCore(engine);
}
