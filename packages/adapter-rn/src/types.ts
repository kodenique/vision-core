export interface PixelDecoder {
  decode(input: RNImageInput, targetWidth: number, targetHeight: number): Promise<RawPixelData>;
}

export type RawPixelData = {
  data: Uint8Array; // RGBA pixel data (4 bytes per pixel)
  width: number;
  height: number;
};

export type RNImageInput =
  | { uri: string }       // remote URL or local file URI
  | { base64: string }    // base64-encoded image
  | { path: string }      // filesystem path
  | number;               // RN require() asset reference
