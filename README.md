# Vision-Core

Backend-first image embedding library. The frontend captures images and sends them to the backend, where all preprocessing and inference happens.

## What is Vision-Core?

Vision-Core converts images into **embedding vectors** — lists of numbers that capture visual features. Two similar images produce similar embeddings, enabling:

- **Similarity Search** — "Find products that look like this photo"
- **Classification** — Compare embeddings against reference categories
- **Clustering** — Group images by visual similarity
- **Face Recognition** — Compare face embeddings to verify identity
- **Duplicate Detection** — Find near-identical images in large datasets

## Architecture

```
Frontend (web/mobile)          Backend (Node.js)
─────────────────────         ──────────────────────────────
Capture image ──── send bytes ──→ Decode (e.g. sharp)
                                    │
                                    ▼
                                  VisionCore.embed(image)
                                    ├─ Resize (bilinear interpolation)
                                    ├─ Normalize (mean/std per channel)
                                    └─ Engine inference (ONNX Runtime)
                                    │
                                    ▼
                                  EmbeddingResult
                                    ├─ embedding: Float32Array
                                    ├─ dimensions: number
                                    └─ modelId: string
```

All image processing runs on the backend. The frontend only needs to capture and send the image.

## Packages

| Package | Description |
|---------|-------------|
| `@vision-core/types` | Shared type definitions (`ImageInput`, `EmbeddingEngine`, `ModelConfig`, etc.) |
| `@vision-core/core` | `VisionCore` class — preprocessing + inference orchestration |
| `@vision-core/engine-onnx` | ONNX Runtime embedding engine implementation |

## Quick Start

```typescript
import { createVisionCore } from '@vision-core/core';
import { createOnnxEngine } from '@vision-core/engine-onnx';
import type { ImageInput, ModelConfig } from '@vision-core/types';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import fs from 'fs/promises';

// 1. Configure the model
const config: ModelConfig = {
  modelSource: './models/mobilenet.onnx',
  modelLoader: (source) => fs.readFile(source).then(b => b.buffer),
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 224,
  inputHeight: 224,
  channels: 3,
  channelOrder: 'CHW',
  normalization: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
};

// 2. Create engine and vision core
const engine = createOnnxEngine(ort);
const vc = createVisionCore(engine);
await vc.initialize(config);

// 3. Receive image from frontend, decode to RGBA pixels
const imageBuffer = await receiveImageFromFrontend(); // however your API receives it
const { data, info } = await sharp(imageBuffer)
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });

const image: ImageInput = {
  data: new Uint8Array(data.buffer),
  width: info.width,
  height: info.height,
};

// 4. Get embedding
const result = await vc.embed(image);
console.log(result.embedding);  // Float32Array of the embedding vector
console.log(result.dimensions); // e.g. 512

// 5. Clean up when done
await vc.dispose();
```

## How It Works

### Step 1: Frontend sends image bytes

The frontend (web browser, React Native app, etc.) captures an image and sends the raw bytes (JPEG, PNG, etc.) to your backend API.

### Step 2: Backend decodes to RGBA pixels

Use a library like `sharp` (Node.js) to decode the image into raw RGBA pixel data:

```typescript
const { data, info } = await sharp(jpegBuffer)
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });

const image: ImageInput = {
  data: new Uint8Array(data.buffer),
  width: info.width,
  height: info.height,
};
```

### Step 3: VisionCore preprocesses and runs inference

When you call `vc.embed(image)`, VisionCore:

1. **Resizes** the image to the model's expected dimensions (e.g. 224x224) using bilinear interpolation
2. **Normalizes** pixel values: converts 0-255 to 0-1, then applies per-channel mean/std normalization
3. **Arranges** data in the model's expected channel order (CHW or HWC)
4. **Runs inference** through the ONNX engine to produce the embedding vector

### Step 4: Use the embedding

The returned `EmbeddingResult` contains:
- `embedding` — `Float32Array` of the raw embedding vector
- `dimensions` — length of the embedding
- `modelId` — identifier of the model that produced it

Use `l2Normalize` from `@vision-core/types` before comparing embeddings:

```typescript
import { l2Normalize } from '@vision-core/types';

const normalized = l2Normalize(result.embedding);
```

## ModelConfig Reference

| Field | Type | Description |
|-------|------|-------------|
| `modelSource` | `string` | Path or URL to the ONNX model file |
| `modelLoader` | `(source: string) => Promise<ArrayBuffer>` | Function to load the model binary |
| `inputTensorName` | `string` | Name of the model's input tensor (check with Netron) |
| `outputTensorName` | `string` | Name of the model's output tensor |
| `inputWidth` | `number` | Expected input width in pixels (e.g. 224) |
| `inputHeight` | `number` | Expected input height in pixels (e.g. 224) |
| `channels` | `3` | Always 3 (RGB) |
| `channelOrder` | `'CHW' \| 'HWC'` | Channel layout — most models use CHW |
| `normalization.mean` | `[number, number, number]` | Per-channel mean for normalization |
| `normalization.std` | `[number, number, number]` | Per-channel std for normalization |

### Common Model Configs

**MobileNetV3 / EfficientNet (ImageNet)**
```typescript
{ mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
// inputWidth: 224, inputHeight: 224, channelOrder: 'CHW'
```

**CLIP (OpenAI)**
```typescript
{ mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711] }
// inputWidth: 224, inputHeight: 224, channelOrder: 'CHW'
```

**No normalization (raw 0-1)**
```typescript
{ mean: [0, 0, 0], std: [1, 1, 1] }
```

## Error Handling

| Error | When |
|-------|------|
| `EngineNotInitializedError` | `embed()` called before `initialize()` |
| `InferenceError` | ONNX engine fails during inference |
| `InvalidInputError` | Tensor validation fails (wrong shape/size) |

## Development

```bash
# Install dependencies
yarn install

# Build all packages
yarn build

# Run all tests
yarn test

# Type check
yarn lint
```
