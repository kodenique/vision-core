/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFiles: ['jest-canvas-mock'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  moduleNameMapper: {
    '^@vision-core/types$': '<rootDir>/../types/src/index.ts',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        tsconfig: {
          strict: true,
          module: 'commonjs',
          moduleResolution: 'node',
          esModuleInterop: true,
          types: ['jest'],
        },
      },
    ],
  },
};
