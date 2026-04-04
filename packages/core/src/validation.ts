import type { TensorInput } from '@vision-core/types';
import { InvalidInputError } from './errors.js';

export function validateTensorInput(input: TensorInput): void {
  if (!input || !(input.data instanceof Float32Array)) {
    throw new InvalidInputError('TensorInput.data must be a Float32Array');
  }
  if (!Array.isArray(input.shape) || input.shape.length === 0) {
    throw new InvalidInputError('TensorInput.shape must be a non-empty array');
  }
  const expectedLength = input.shape.reduce((a, b) => a * b, 1);
  if (input.data.length !== expectedLength) {
    throw new InvalidInputError(
      `TensorInput.data length (${input.data.length}) does not match shape product (${expectedLength})`
    );
  }
}
