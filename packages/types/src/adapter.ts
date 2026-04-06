/**
 * Raw RGBA pixel data received from the frontend.
 * The backend decodes the image (e.g. via sharp) and passes this to VisionCore.
 */
export type ImageInput = {
  data: Uint8Array; // RGBA pixel data (4 bytes per pixel)
  width: number;
  height: number;
};
