/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  moduleNameMapper: {
    '^@vision-core/types$': '<rootDir>/../types/src/index.ts',
    '^@vision-core/core$': '<rootDir>/../core/src/index.ts',
    '^@vision-core/engine-onnx$': '<rootDir>/../engine-onnx/src/index.ts',
    '^@vision-core/adapter-web$': '<rootDir>/../adapter-web/src/index.ts',
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { useESM: false, tsconfig: 'tsconfig.json' }],
  },
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
};
