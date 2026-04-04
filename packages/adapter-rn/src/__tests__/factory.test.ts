import { createRNAdapter } from '../factory.js';
import { RNImageAdapter } from '../rn-adapter.js';
import type { PixelDecoder, RawPixelData } from '../types.js';

const mockDecoder: PixelDecoder = {
  decode(_input, targetWidth, targetHeight): Promise<RawPixelData> {
    const data = new Uint8Array(targetWidth * targetHeight * 4);
    return Promise.resolve({ data, width: targetWidth, height: targetHeight });
  },
};

const config = {
  normalization: {
    mean: [0.485, 0.456, 0.406] as [number, number, number],
    std: [0.229, 0.224, 0.225] as [number, number, number],
  },
  channelOrder: 'CHW' as const,
};

describe('createRNAdapter', () => {
  it('returns an RNImageAdapter instance', () => {
    const adapter = createRNAdapter(mockDecoder, config);
    expect(adapter).toBeInstanceOf(RNImageAdapter);
  });

  it('returned adapter has a preprocess method (implements ImageAdapter)', () => {
    const adapter = createRNAdapter(mockDecoder, config);
    expect(typeof adapter.preprocess).toBe('function');
  });

  it('preprocess resolves to a TensorInput with correct shape', async () => {
    const adapter = createRNAdapter(mockDecoder, config);
    const result = await adapter.preprocess({ uri: 'test.jpg' }, { width: 4, height: 4 });

    expect(result).toHaveProperty('data');
    expect(result).toHaveProperty('shape');
    expect(result.data).toBeInstanceOf(Float32Array);
    expect(result.shape).toEqual([3, 4, 4]);
  });
});
