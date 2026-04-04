export type ImageSize = {
  width: number;
  height: number;
};

export type ModelConfig = {
  modelSource: string;
  modelLoader: (source: string) => Promise<ArrayBuffer>;
  inputTensorName: string;
  outputTensorName: string;
  inputWidth: number;
  inputHeight: number;
  channels: 3;
  channelOrder: 'CHW' | 'HWC';
  normalization: {
    mean: [number, number, number];
    std: [number, number, number];
  };
};
