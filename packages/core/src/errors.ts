export class EngineNotInitializedError extends Error {
  constructor(message = 'Engine has not been initialized. Call initialize() first.') {
    super(message);
    this.name = 'EngineNotInitializedError';
  }
}

export class InvalidInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidInputError';
  }
}

export class InferenceError extends Error {
  constructor(message: string, public readonly cause?: unknown) {
    super(message);
    this.name = 'InferenceError';
  }
}

export class AdapterError extends Error {
  constructor(message: string, public readonly cause?: unknown) {
    super(message);
    this.name = 'AdapterError';
  }
}
