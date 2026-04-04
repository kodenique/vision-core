# Vision-Core: Complete Beginner's Guide

Welcome! If you've never worked with image embeddings, ONNX models, or computer vision pipelines before, this guide is for you. We'll explain everything from the ground up, with concrete examples and use cases.

---

## Table of Contents

1. [What is Vision-Core?](#what-is-vision-core)
2. [How the Pipeline Works (Step by Step)](#how-the-pipeline-works-step-by-step)
3. [ModelConfig — Every Parameter Explained](#modelconfig--every-parameter-explained)
4. [Web Adapter — Complete Guide](#web-adapter--complete-guide)
5. [React Native Adapter — Complete Guide](#react-native-adapter--complete-guide)
6. [ONNX Engine — Complete Guide](#onnx-engine--complete-guide)
7. [l2Normalize — When and Why](#l2normalize--when-and-why)
8. [Real-World Use Cases with Full Code Examples](#real-world-use-cases-with-full-code-examples)
9. [How to Use Your Own Trained Model](#how-to-use-your-own-trained-model)
10. [Common Models and Their Configs](#common-models-and-their-configs)
11. [Error Handling](#error-handling)
12. [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## What is Vision-Core?

### In Plain English

Vision-Core converts images into numbers. More precisely, it takes an image (a photo, a screenshot, whatever you have) and outputs a **fingerprint** — a list of numbers called an **embedding vector** — that captures the visual features of that image.

Think of it like this:
- **Without Vision-Core**: You have a picture of a cat. It's just pixels.
- **With Vision-Core**: You have a picture of a cat. Vision-Core says: "This image can be represented as [0.234, -0.156, 0.891, ...]" — a list of 512 (or more) numbers that describe what makes this cat image unique.

Why is this useful? Because two cat images will have *similar* numbers, and a cat image will have *very different* numbers from a dog image. This lets you do powerful things without needing to understand the image yourself.

### Why Embeddings Matter

Embeddings unlock five super-powers:

#### 1. **Similarity Search** ("Find similar items")
Store embeddings of 1 million product photos. A customer uploads a photo: "I want jeans like this." Vision-Core embeds their photo, and you search for the most similar embeddings in your database. Instant visual search.

#### 2. **Classification** ("What category is this?")
Instead of training a custom neural network, embed reference images for each class (shirts, pants, shoes), then embed a new image and see which class it's closest to. Works even with classes the model has never seen before (zero-shot classification).

#### 3. **Clustering** ("Group these images together")
Run Vision-Core on 10,000 product photos. Group them by embedding similarity, and suddenly you have natural clusters: dresses here, jackets there. No manual labeling needed.

#### 4. **Face Recognition** ("Is this the same person?")
Embed face photo A, embed face photo B, compare the embeddings. If they're similar enough, it's the same person. Different enough? Different person. No passwords needed.

#### 5. **Visual Quality Control** ("Is this image acceptable?")
Embed reference images of good products, embed the current production photo, compare. If dissimilar, something's wrong with production. Instant defect detection.

### Real-World Examples

**E-commerce**: Pinterest, Google Lens, Amazon Visual Search — all use image embeddings to find similar items.

**Facial Recognition**: Your phone unlocks via face recognition. That's embeddings in action.

**Content Moderation**: Social media platforms embed uploaded images and compare them against a database of known problematic images.

**Medical Imaging**: Radiologists compare a patient's scan against past cases. Embeddings make this search instant.

**Manufacturing**: Quality control: embed reference images of perfect products, compare against the production line in real-time.

### The Architecture

Here's what Vision-Core does, visually:

```
┌─────────────┐
│   Your      │
│   Image     │
│ (file/URL)  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│  ImageAdapter                    │
│  (Preprocess)                    │
│                                  │
│  1. Load image from blob/URL     │
│  2. Resize to model size (224×224)
│  3. Normalize pixels (0-1, apply mean/std)
│  4. Arrange into tensor shape    │
└──────┬───────────────────────────┘
       │
       │ TensorInput: Float32Array + shape
       │
       ▼
┌──────────────────────────────────┐
│  EmbeddingEngine (ONNX)          │
│  (Inference)                     │
│                                  │
│  Runs the ML model forward pass  │
│  Input: normalized tensor        │
│  Output: embedding vector        │
└──────┬───────────────────────────┘
       │
       │ EmbeddingResult: Float32Array
       │
       ▼
┌─────────────────────────────────┐
│  Your embedding vector          │
│  [0.234, -0.156, 0.891, ...]   │
│  (512 floats, or however many   │
│   the model outputs)            │
└─────────────────────────────────┘
       │
       │ Optional: l2Normalize
       │ for cosine similarity
       │
       ▼
┌──────────────────────────────────┐
│  Normalized embedding vector     │
│  (magnitude = 1.0)               │
│  Ready for comparison            │
└──────────────────────────────────┘
```

---

## How the Pipeline Works (Step by Step)

Let's walk through the entire journey from raw image to embedding vector, understanding every detail.

### Step 1: You Provide an Image

This can be:
- A file from a form input (`<input type="file">`)
- A URL to an image on the internet
- A `Blob` or `File` object
- An `HTMLImageElement` (if you've already loaded it)
- An `ImageBitmap` (pre-loaded by the browser)
- In React Native: a URI, base64 string, or file path

Vision-Core accepts all of these because the **ImageAdapter** handles the conversion.

### Step 2: The Adapter Loads and Resizes

The adapter's job: take your image (any size, any format) and convert it into exactly what the model expects.

**Real example**: You upload a 3000×2000 photo, but the model expects 224×224.

```
Original: 3000×2000 pixels
         ↓
         Resize to 224×224
         ↓
Resized: 224×224 pixels
```

On the web, this uses the **Canvas API** — we draw the image onto a canvas of size 224×224, and the browser automatically handles the resizing (bilinear interpolation).

In React Native, the adapter uses a **PixelDecoder** (you provide this, because RN doesn't have Canvas). The decoder uses whatever image library you prefer (expo-image-manipulator, react-native-image-crop-picker, etc.) to resize the image.

### Step 3: Normalize the Pixels

**Problem**: Raw pixel values are 0-255. Different images have different brightness, contrast, color distributions. The model was trained on normalized data, so we need to match that.

**Solution**: Normalization.

```
Raw pixel:  255 (pure white)
           ↓
Divide by 255:  1.0
           ↓
Subtract mean (e.g., 0.485 for red):  1.0 - 0.485 = 0.515
           ↓
Divide by std (e.g., 0.229 for red):  0.515 / 0.229 ≈ 2.249
           ↓
Normalized value:  2.249
```

**Why mean and std?** The model was trained on ImageNet, a dataset with a specific distribution. Subtracting the mean "centers" the data, and dividing by std "scales" it. This tells the model: "This image is from the same distribution I was trained on."

**Different models, different values**:
- Most image models use ImageNet values: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- CLIP models use: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
- YOLO models might use: mean=[0, 0, 0], std=[1, 1, 1] (no normalization)

We apply this per-channel (red, green, blue separately).

### Step 4: Arrange Pixels into a Tensor

Now we have normalized floats. But the model expects them in a specific **memory layout**. There are two common formats:

#### **CHW (Channels-Height-Width)**
Used by PyTorch. Organize by channel first:

```
Imagine a 224×224 image with R, G, B channels:

Memory layout for CHW:
┌────────────────────────────────────┐
│  All Red pixels (224×224 = 50,176) │
│  R[0,0], R[0,1], R[0,2], ..., R[223,223]
├────────────────────────────────────┤
│  All Green pixels (50,176)         │
│  G[0,0], G[0,1], G[0,2], ..., G[223,223]
├────────────────────────────────────┤
│  All Blue pixels (50,176)          │
│  B[0,0], B[0,1], B[0,2], ..., B[223,223]
└────────────────────────────────────┘

Total: 224 × 224 × 3 = 150,528 floats

Shape: [3, 224, 224]
```

#### **HWC (Height-Width-Channels)**
Used by TensorFlow. Organize by position first:

```
Memory layout for HWC:
┌──────────────────────────────────────┐
│  Pixel [0,0] RGB (3 floats)          │
│  R[0,0], G[0,0], B[0,0]              │
├──────────────────────────────────────┤
│  Pixel [0,1] RGB (3 floats)          │
│  R[0,1], G[0,1], B[0,1]              │
├──────────────────────────────────────┤
│  ... repeat 224×224 times ...        │
└──────────────────────────────────────┘

Total: 224 × 224 × 3 = 150,528 floats

Shape: [224, 224, 3]
```

**Same data, different order.** The model doesn't care which you use — you just need to use the right one for *your* model.

**How do you know which one?** Use the **Netron tool**: https://netron.app/

Upload your `.onnx` file, look at the input node's shape. If it says `[1, 3, 224, 224]`, that's `CHW` (the `1` is batch size). If it says `[1, 224, 224, 3]`, that's `HWC`.

### Step 5: Run Inference

The tensor goes into the ONNX engine. The engine:
1. Takes the normalized tensor
2. Feeds it to the loaded model
3. Runs the model's forward pass (all the matrix multiplications and activations)
4. Returns the output

This is where the magic happens — the model has learned, from millions of images, how to extract meaningful features. You don't need to understand the internal math. You just trust that it works.

**Output**: Another tensor, usually 512 or 1024 floats. This is your **embedding**.

### Step 6: You Get the Embedding Vector

The result is an `EmbeddingResult`:

```typescript
{
  embedding: Float32Array([0.234, -0.156, 0.891, ...]),  // 512 floats
  dimensions: 512,                                        // size of embedding
  modelId: "path/to/model.onnx"                           // which model made it
}
```

### Step 7: (Optional) Normalize the Embedding

For some tasks (especially similarity search), you might want to **l2-normalize** the embedding. This makes all embeddings the same "magnitude" (length), so you can compare them fairly.

```
Original embedding: [0.234, -0.156, 0.891]
Magnitude: √(0.234² + 0.156² + 0.891²) ≈ 0.925

L2-normalized: [0.234/0.925, -0.156/0.925, 0.891/0.925]
             = [0.253, -0.169, 0.963]

Magnitude after normalization: 1.0 (exactly)
```

Now two embeddings can be compared with cosine similarity, which is just a dot product:

```typescript
cosine_similarity = embedding1.dot(embedding2)
// Result: -1.0 to 1.0
// 1.0 = identical
// 0.0 = orthogonal
// -1.0 = opposite
```

---

## ModelConfig — Every Parameter Explained

The `ModelConfig` tells Vision-Core everything it needs to know about your model. Let's break down every field with real examples.

```typescript
type ModelConfig = {
  modelSource: string;                              // path or URL to .onnx
  modelLoader: (source: string) => Promise<ArrayBuffer>; // how to fetch bytes
  inputTensorName: string;                          // e.g., "input"
  outputTensorName: string;                         // e.g., "output"
  inputWidth: number;                               // e.g., 224
  inputHeight: number;                              // e.g., 224
  channels: 3;                                      // always RGB
  channelOrder: 'CHW' | 'HWC';                     // tensor layout
  normalization: {
    mean: [number, number, number];                 // per-channel mean
    std: [number, number, number];                  // per-channel std
  };
};
```

### modelSource: string

**What it is**: A string pointing to your model file.

**Examples**:
- `"./models/mobilenetv3.onnx"` — local file path
- `"/assets/embeddings-model.onnx"` — bundled in your app
- `"https://example.com/models/resnet50.onnx"` — URL to remote server

This is just a string identifier. Vision-Core passes it to `modelLoader`, which decides how to actually fetch the file.

### modelLoader: (source: string) => Promise<ArrayBuffer>

**What it is**: A callback function you provide that loads the model bytes.

**Why a callback?** Different platforms load files differently:
- **Browser**: `fetch(modelSource).then(r => r.arrayBuffer())`
- **React Native**: Native file system APIs
- **Node.js**: `fs.readFile()`
- **Bundled**: Import as data URI or base64

**Examples**:

**Browser (fetch from URL)**:
```typescript
const modelLoader = async (source: string) => {
  const response = await fetch(source);
  return response.arrayBuffer();
};
```

**Browser (bundled asset)**:
```typescript
import modelBase64 from './models/mobilenetv3.onnx?base64';

const modelLoader = async (source: string) => {
  // Convert base64 to ArrayBuffer
  const binaryString = atob(modelBase64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};
```

**React Native (file system)**:
```typescript
import * as FileSystem from 'expo-file-system';

const modelLoader = async (source: string) => {
  const base64 = await FileSystem.readAsStringAsync(source, {
    encoding: FileSystem.EncodingType.Base64,
  });
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};
```

### inputTensorName: string

**What it is**: The name of the input node in your ONNX model.

**How to find it**: Use Netron (https://netron.app/):
1. Upload your `.onnx` file
2. Look at the leftmost node — the input
3. Find its name (usually "input", "images", "data", etc.)

**Examples**:
- `"input"` — common default
- `"images"` — if the model expects image data
- `"pixel_values"` — used by some CLIP models
- `"data"` — older models

If you get this wrong, you'll get an error like "Input tensor 'wrong_name' not found."

### outputTensorName: string

**What it is**: The name of the output node that contains the embedding.

**How to find it**: Same as input — use Netron, look at the rightmost node.

**Examples**:
- `"output"` — common default
- `"embeddings"` — if the model explicitly names it
- `"logits"` — if you're doing classification and want the logits
- `"features"` — common for feature extractors

**Important**: Some models have multiple outputs (e.g., bounding boxes AND confidence scores). You want the one that's the embedding/features. Netron shows you all outputs; pick the right one.

### inputWidth: number and inputHeight: number

**What it is**: The dimensions the model expects, in pixels.

**Common values**:
- **MobileNetV3**: 224×224 (lightweight, fast)
- **ResNet-50**: 224×224 (standard)
- **EfficientNet**: varies, but often 224×224, 256×256, or 380×380
- **CLIP**: 224×224
- **YOLO**: 640×640
- **Face models (ArcFace)**: 112×112

These are defined when the model is trained. You can't change them. If you give the model a different size, it will produce garbage.

**How to find it?** Netron shows the input shape. For example, `[1, 3, 224, 224]` means 224×224 (the `1` and `3` are batch size and channels).

### channels: 3

**Always 3.** This means RGB (red, green, blue).

**Why not 4?** The 4th channel would be alpha (transparency). Most models don't care about transparency, so we discard it. Vision-Core converts all images to RGB before processing.

### channelOrder: 'CHW' | 'HWC'

**CHW** (Channels-Height-Width):
- PyTorch convention
- Used by most ONNX models from PyTorch
- Shape: `[3, 224, 224]`

**HWC** (Height-Width-Channels):
- TensorFlow convention
- Some TensorFlow ONNX exports use this
- Shape: `[224, 224, 3]`

**How to know?** Look at the input shape in Netron:
- `[1, 3, 224, 224]` → CHW
- `[1, 224, 224, 3]` → HWC

If you get this wrong, the model will process the image incorrectly, and embeddings will be garbage.

### normalization: { mean, std }

**What it is**: Per-channel mean and standard deviation used during model training.

**ImageNet values** (used by most pre-trained models):
```typescript
normalization: {
  mean: [0.485, 0.456, 0.406],    // for R, G, B
  std: [0.229, 0.224, 0.225],
}
```

**CLIP ViT-B/32**:
```typescript
normalization: {
  mean: [0.48145466, 0.4578275, 0.40821073],
  std: [0.26862954, 0.26130258, 0.27577711],
}
```

**No normalization** (some custom models):
```typescript
normalization: {
  mean: [0, 0, 0],
  std: [1, 1, 1],  // dividing by 1 = no effect
}
```

**Why these specific values?** They're the mean and standard deviation of the training dataset. When you subtract the mean, you're saying "center the data." When you divide by std, you're saying "scale it consistently."

**How to find them?** Check the model's documentation or training code. For popular models, I've included a table below.

---

## Web Adapter — Complete Guide

The web adapter runs in browsers. It uses the **Canvas API** under the hood to resize and process images.

### Supported Input Types

```typescript
type WebImageInput =
  | File                  // from <input type="file">
  | Blob                  // from fetch(), canvas, etc.
  | HTMLImageElement      // <img> tag already loaded
  | ImageBitmap           // pre-loaded by browser
  | string;               // URL to fetch
```

### When to Use Each

**File** (from file input):
```html
<input type="file" id="imageFile">
```
```typescript
const input = document.getElementById('imageFile') as HTMLInputElement;
const file = input.files?.[0];
if (file) {
  const embedding = await visionCore.embed(file);
}
```

**Blob** (from fetch or canvas):
```typescript
// Fetch an image
const response = await fetch('https://example.com/image.jpg');
const blob = await response.blob();
const embedding = await visionCore.embed(blob);
```

**HTMLImageElement** (already in DOM):
```html
<img id="myImage" src="photo.jpg" />
```
```typescript
const img = document.getElementById('myImage') as HTMLImageElement;
const embedding = await visionCore.embed(img);
```

**ImageBitmap** (pre-decoded):
```typescript
const blob = /* ... */;
const bitmap = await createImageBitmap(blob);
const embedding = await visionCore.embed(bitmap);  // fastest
```

**URL string** (fetch and embed):
```typescript
const embedding = await visionCore.embed('https://example.com/image.jpg');
```

### Full Working Example (Web)

```typescript
import { VisionCore } from '@vision-core/core';
import { createOnnxEngine } from '@vision-core/engine-onnx';
import { WebImageAdapter } from '@vision-core/adapter-web';
import * as ort from 'onnxruntime-web';

// Step 1: Create the engine (requires ONNX Runtime)
const engine = createOnnxEngine(ort);

// Step 2: Create the adapter
const adapter = new WebImageAdapter({
  normalization: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
  channelOrder: 'CHW',
});

// Step 3: Create VisionCore
const visionCore = new VisionCore(engine, adapter);

// Step 4: Initialize with model config
await visionCore.initialize({
  modelSource: './models/mobilenetv3.onnx',
  modelLoader: async (source: string) => {
    const response = await fetch(source);
    return response.arrayBuffer();
  },
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
});

// Step 5: Embed an image
const fileInput = document.getElementById('imageFile') as HTMLInputElement;
const file = fileInput.files?.[0];
if (file) {
  const result = await visionCore.embed(file);
  console.log('Embedding:', result.embedding);
  console.log('Dimensions:', result.dimensions);  // e.g., 1280
}

// Step 6: Clean up when done
await visionCore.dispose();
```

### How Resizing Works

Vision-Core uses the Canvas API:

```typescript
const canvas = document.createElement('canvas');
canvas.width = 224;
canvas.height = 224;
const ctx = canvas.getContext('2d');
ctx.drawImage(image, 0, 0, 224, 224);  // browser handles resizing
const imageData = ctx.getImageData(0, 0, 224, 224);
```

The browser's `drawImage()` method handles all the resizing automatically (using bilinear interpolation by default).

### How Normalization Works

```typescript
// Raw pixel from canvas: 0-255 (RGBA)
const rawR = imageData.data[i * 4];      // Red
const rawG = imageData.data[i * 4 + 1];  // Green
const rawB = imageData.data[i * 4 + 2];  // Blue
// Alpha is skipped

// CHW layout: all reds, then all greens, then all blues
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

const normalizedR = (rawR / 255 - mean[0]) / std[0];
const normalizedG = (rawG / 255 - mean[1]) / std[1];
const normalizedB = (rawB / 255 - mean[2]) / std[2];

// Store in CHW order
result[pixelIndex] = normalizedR;                    // red channel
result[numPixels + pixelIndex] = normalizedG;        // green channel
result[numPixels * 2 + pixelIndex] = normalizedB;    // blue channel
```

---

## React Native Adapter — Complete Guide

The React Native adapter is different because RN doesn't have Canvas. Instead, it uses a **PixelDecoder** — a function you provide.

### Why RN Needs a Different Approach

React Native doesn't have:
- Canvas API
- `document.createElement()`
- Built-in image manipulation

So Vision-Core asks: "You provide me a way to decode and resize images, and I'll handle the normalization."

### The PixelDecoder Pattern

You provide a decoder with this interface:

```typescript
interface PixelDecoder {
  decode(input: RNImageInput, targetWidth: number, targetHeight: number): Promise<RawPixelData>;
}

type RawPixelData = {
  data: Uint8Array;  // RGBA pixel data (4 bytes per pixel)
  width: number;
  height: number;
};
```

The decoder takes an input (URI, base64, path, or asset reference) and returns **RGBA pixel bytes** at the target size.

### Supported Input Types

```typescript
type RNImageInput =
  | { uri: string }       // URL or file URI: file:///path/to/image.jpg
  | { base64: string }    // base64-encoded image data
  | { path: string }      // filesystem path
  | number;               // require('./image.jpg') asset reference
```

### When to Use Each

**URI** (most common — remote URL or local file):
```typescript
{ uri: 'https://example.com/image.jpg' }
{ uri: 'file:///storage/emulated/0/image.jpg' }
```

**Base64** (image already decoded to string):
```typescript
{ base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==' }
```

**Path** (filesystem path):
```typescript
{ path: '/data/user/0/com.myapp/cache/image.jpg' }
```

**Asset Reference** (bundled with app):
```typescript
{ uri: require('./images/logo.png') }  // returns a number in older RN versions
// or just:
require('./images/logo.png')  // the number directly
```

### Example 1: Using expo-image-manipulator

```typescript
import * as ImageManipulator from 'expo-image-manipulator';
import type { PixelDecoder, RNImageInput } from '@vision-core/adapter-rn';

const pixelDecoder: PixelDecoder = {
  async decode(input: RNImageInput, targetWidth: number, targetHeight: number) {
    // Step 1: Normalize input to a URI
    let uri: string;
    if (typeof input === 'number') {
      // Asset reference
      uri = Image.resolveAssetSource(input).uri;
    } else if ('uri' in input) {
      uri = input.uri;
    } else if ('base64' in input) {
      // Create a data URI
      uri = `data:image/jpeg;base64,${input.base64}`;
    } else if ('path' in input) {
      uri = `file://${input.path}`;
    } else {
      throw new Error('Invalid input');
    }

    // Step 2: Resize using expo-image-manipulator
    const result = await ImageManipulator.manipulateAsync(uri, [
      { resize: { width: targetWidth, height: targetHeight } },
    ], { format: 'jpeg', compress: 1 });

    // Step 3: Get the resized image as base64
    const manipulatedImage = await fetch(result.uri);
    const arrayBuffer = await manipulatedImage.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);

    // Step 4: Convert JPEG bytes to RGBA pixel data
    // (You'd use a JPEG decoder here — see react-native-image-decoder)
    // For this example, we'll assume you have one:
    const pixelData = await jpegDecoder.decode(uint8Array);

    return {
      data: pixelData,  // Uint8Array of RGBA bytes
      width: targetWidth,
      height: targetHeight,
    };
  },
};
```

### Example 2: Using react-native-image-crop-picker

```typescript
import ImagePicker from 'react-native-image-crop-picker';
import type { PixelDecoder } from '@vision-core/adapter-rn';

const pixelDecoder: PixelDecoder = {
  async decode(input, targetWidth, targetHeight) {
    // image-crop-picker can resize for you
    const image = await ImagePicker.openPicker({
      width: targetWidth,
      height: targetHeight,
      cropping: true,
    });

    // image.data is base64
    const pixelData = base64ToPixels(image.data);

    return {
      data: pixelData,
      width: targetWidth,
      height: targetHeight,
    };
  },
};

function base64ToPixels(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}
```

### The Built-in Bilinear Resize Fallback

If your decoder returns the wrong dimensions, Vision-Core has a fallback:

```typescript
// If decoder returned 300×300 but we need 224×224,
// Vision-Core automatically resizes using bilinear interpolation
if (pixelData.width !== targetWidth || pixelData.height !== targetHeight) {
  pixelData = resizePixelData(pixelData, targetWidth, targetHeight);
}
```

Bilinear interpolation means:
1. For each output pixel, find the corresponding location in the input image
2. Look at the 4 surrounding input pixels
3. Blend them together based on distance
4. Result: smooth resizing without pixelation

### Full Working Example (React Native)

```typescript
import { Image } from 'react-native';
import { VisionCore } from '@vision-core/core';
import { createRNAdapter } from '@vision-core/adapter-rn';
import { createOnnxEngine } from '@vision-core/engine-onnx';
import * as ort from 'onnxruntime-react-native';
import * as FileSystem from 'expo-file-system';
import type { PixelDecoder, RNImageInput } from '@vision-core/adapter-rn';

// Step 1: Define your pixel decoder (using expo-image-manipulator example)
const pixelDecoder: PixelDecoder = {
  async decode(input: RNImageInput, targetWidth: number, targetHeight: number) {
    // ... (implementation from examples above)
    return { data: Uint8Array, width: targetWidth, height: targetHeight };
  },
};

// Step 2: Create engine
const engine = createOnnxEngine(ort);

// Step 3: Create adapter
const adapter = createRNAdapter(pixelDecoder, {
  normalization: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
  channelOrder: 'CHW',
});

// Step 4: Create VisionCore
const visionCore = new VisionCore(engine, adapter);

// Step 5: Initialize
await visionCore.initialize({
  modelSource: 'models/mobilenetv3.onnx',
  modelLoader: async (source: string) => {
    // Load from app bundle or file system
    const base64 = await FileSystem.readAsStringAsync(source, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  },
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
});

// Step 6: Embed an image from camera or gallery
const embedding = await visionCore.embed({ uri: 'file:///path/to/photo.jpg' });
console.log('Embedding:', embedding.embedding);

// Step 7: Clean up
await visionCore.dispose();
```

---

## ONNX Engine — Complete Guide

The engine runs the actual ML model. It uses **ONNX Runtime**, a library that runs ONNX models efficiently.

### What is ONNX?

**ONNX** = Open Neural Network Exchange.

It's a standard format for machine learning models. Instead of exporting a PyTorch model as `.pt` and a TensorFlow model as `.pb`, you export both as `.onnx`, and they're compatible with the same runtime.

Think of it like the MP4 format for video — different cameras create MP4s, different players read MP4s, and they all work together.

### Why Consumer-Provided Runtime?

Vision-Core asks you to provide the ONNX Runtime because:

1. **Different platforms, different runtimes**
   - Web: `onnxruntime-web` (WebAssembly + WebGL)
   - React Native: `onnxruntime-react-native` (native code)
   - Node.js: `onnxruntime-node` (native bindings)

2. **File size and performance optimization**
   - You can choose backends (WebGL, CUDA, CoreML, etc.)
   - Some platforms need special builds

3. **Flexibility**
   - You control versioning
   - You can swap out the runtime if needed

Vision-Core doesn't care which runtime you use — it just needs the interface:

```typescript
interface OnnxRuntime {
  InferenceSession: {
    create(data: ArrayBuffer, options?: unknown): Promise<OnnxInferenceSession>;
  };
}

interface OnnxInferenceSession {
  run(feeds: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>>;
  release(): Promise<void>;
}
```

### Setting Up ONNX Runtime for Web

```bash
npm install onnxruntime-web
```

```typescript
import * as ort from 'onnxruntime-web';

const engine = createOnnxEngine(ort);

// Optional: configure backend
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
ort.env.wasm.numThreads = 4;
```

### Setting Up ONNX Runtime for React Native

```bash
npm install onnxruntime-react-native
```

```typescript
import * as ort from 'onnxruntime-react-native';

const engine = createOnnxEngine(ort);
```

### Setting Up ONNX Runtime for Node.js

```bash
npm install onnxruntime-node
```

```typescript
import * as ort from 'onnxruntime-node';

const engine = createOnnxEngine(ort);
```

### How Model Loading Works

```typescript
// Your ModelConfig
const config = {
  modelSource: './models/mobilenetv3.onnx',
  modelLoader: async (source: string) => {
    const response = await fetch(source);
    return response.arrayBuffer();  // ArrayBuffer of ONNX file bytes
  },
  // ... other config
};

// When you call initialize():
await visionCore.initialize(config);

// Behind the scenes:
// 1. Engine calls: config.modelLoader(config.modelSource)
// 2. modelLoader fetches and returns ArrayBuffer
// 3. Engine passes it to: ort.InferenceSession.create(arrayBuffer)
// 4. ONNX Runtime deserializes the model and prepares it for inference
// 5. Session is stored internally
```

### How Inference Works

```typescript
// You have a tensor ready:
const tensorInput: TensorInput = {
  data: Float32Array([0.1, 0.2, ...]),  // normalized pixels
  shape: [3, 224, 224],                  // CHW
};

// Engine feeds it to the model:
const feeds: Record<string, { data: Float32Array; dims: number[] }> = {
  'input': { data: tensorInput.data, dims: tensorInput.shape },
};

const results = await session.run(feeds);

// results contains all output tensors:
// { 'output': { data: Float32Array([...]), dims: [1, 1280] } }

const outputTensor = results[config.outputTensorName];
const embedding = outputTensor.data instanceof Float32Array
  ? outputTensor.data
  : new Float32Array(outputTensor.data);
```

---

## l2Normalize — When and Why

L2 normalization makes an embedding vector have a magnitude (length) of exactly 1.0.

### The Math

```typescript
function l2Normalize(embedding: Float32Array): Float32Array {
  // Step 1: Calculate magnitude (L2 norm)
  let sumOfSquares = 0;
  for (let i = 0; i < embedding.length; i++) {
    sumOfSquares += embedding[i] * embedding[i];
  }
  const magnitude = Math.sqrt(sumOfSquares);

  // Step 2: Divide each element by magnitude
  const normalized = new Float32Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) {
    normalized[i] = embedding[i] / magnitude;
  }
  return normalized;
}
```

### Simple Example

```
Original: [3, 4]
Magnitude: √(3² + 4²) = √(9 + 16) = √25 = 5

Normalized: [3/5, 4/5] = [0.6, 0.8]
Check: √(0.6² + 0.8²) = √(0.36 + 0.64) = √1 = 1 ✓
```

### When to Use L2Normalize

**Use it when**: Comparing embeddings with cosine similarity.

```typescript
const embedding1 = await visionCore.embed(image1);
const embedding2 = await visionCore.embed(image2);

// Normalize both
const norm1 = l2Normalize(embedding1.embedding);
const norm2 = l2Normalize(embedding2.embedding);

// Cosine similarity = dot product of normalized vectors
let similarity = 0;
for (let i = 0; i < norm1.length; i++) {
  similarity += norm1[i] * norm2[i];
}

// similarity is now -1.0 to 1.0
// 1.0 = identical images
// 0.0 = completely unrelated
// -1.0 = opposite
```

**Don't use it when**:
- Your model already normalizes embeddings internally (e.g., some face recognition models)
- You're using Euclidean distance (`sqrt((a-b)² + (c-d)² + ...)`)
- You're using embeddings for clustering algorithms that don't expect normalized vectors

### Why Normalization Helps with Similarity

Without normalization:
- An embedding with large values might seem "more similar" just because of magnitude
- A long vector and short vector measuring the same direction would have different similarity scores

With normalization:
- All embeddings have the same magnitude (1.0)
- Direction is the only thing that matters
- Cosine similarity is fair and interpretable

---

## Real-World Use Cases with Full Code Examples

### 8.1 Image Similarity Search (E-commerce)

**Goal**: Store embeddings of 10,000 products. User uploads a photo. Find the 10 most similar products.

**Setup**:

```typescript
import { VisionCore } from '@vision-core/core';
import { createOnnxEngine } from '@vision-core/engine-onnx';
import { WebImageAdapter } from '@vision-core/adapter-web';
import * as ort from 'onnxruntime-web';
import { l2Normalize } from '@vision-core/types';

// Initialize VisionCore
const engine = createOnnxEngine(ort);
const adapter = new WebImageAdapter({
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
  channelOrder: 'CHW',
});
const visionCore = new VisionCore(engine, adapter);

await visionCore.initialize({
  modelSource: './models/mobilenetv3.onnx',
  modelLoader: async (source) => {
    const response = await fetch(source);
    return response.arrayBuffer();
  },
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 224,
  inputHeight: 224,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
});

// Product database
interface Product {
  id: string;
  name: string;
  imageUrl: string;
  embedding: Float32Array;
}

const products: Product[] = [];

// Step 1: Build the database (do this once, in background)
async function buildProductDatabase() {
  const productList = [
    { id: '1', name: 'Blue Jeans', imageUrl: '/products/jeans-blue.jpg' },
    { id: '2', name: 'Red Dress', imageUrl: '/products/dress-red.jpg' },
    // ... 9,998 more products
  ];

  for (const product of productList) {
    const result = await visionCore.embed(product.imageUrl);
    products.push({
      ...product,
      embedding: l2Normalize(result.embedding),
    });
  }

  console.log(`Indexed ${products.length} products`);
}

// Step 2: User uploads a photo and searches
async function searchSimilarProducts(userImageFile: File): Promise<Product[]> {
  const userResult = await visionCore.embed(userImageFile);
  const userEmbedding = l2Normalize(userResult.embedding);

  // Calculate cosine similarity with each product
  const similarities = products.map((product) => {
    let similarity = 0;
    for (let i = 0; i < userEmbedding.length; i++) {
      similarity += userEmbedding[i] * product.embedding[i];
    }
    return { product, similarity };
  });

  // Sort by similarity, return top 10
  return similarities
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 10)
    .map(({ product }) => product);
}

// Usage:
const fileInput = document.getElementById('searchInput') as HTMLInputElement;
fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    const results = await searchSimilarProducts(file);
    console.log('Top 10 similar products:', results.map((p) => p.name));
  }
});
```

### 8.2 Using with YOLO Models

**Scenario**: You have a YOLO model that detects objects (e.g., clothing items in images). You want to extract embeddings of each detected item.

**Important note**: YOLO outputs bounding boxes and confidence scores, NOT embeddings. You need two models:
1. YOLO for detection
2. Vision-Core for feature extraction of detected regions

**Setup**:

```typescript
// Model 1: YOLO v8 for detection (returns bounding boxes)
const yoloConfig = {
  modelSource: './models/yolov8n.onnx',
  modelLoader: async (source) => {
    const response = await fetch(source);
    return response.arrayBuffer();
  },
  inputTensorName: 'images',
  outputTensorName: 'output0',
  inputWidth: 640,
  inputHeight: 640,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0, 0, 0], std: [1, 1, 1] },  // YOLO uses no normalization
};

// Model 2: MobileNetV3 for embeddings (separate VisionCore)
const visionCore = new VisionCore(engine, adapter);
await visionCore.initialize({
  modelSource: './models/mobilenetv3.onnx',
  modelLoader: async (source) => {
    const response = await fetch(source);
    return response.arrayBuffer();
  },
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 224,
  inputHeight: 224,
  channels: 3,
  channelOrder: 'CHW',
  normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
});

// Usage:
async function detectAndEmbed(imageUrl: string) {
  // Step 1: Load image
  const image = await createImageBitmap(await fetch(imageUrl).then((r) => r.blob()));

  // Step 2: Run YOLO detection
  // (You'd use a separate YOLO library or run YOLO inference manually)
  const detections = await yoloModel.detect(image);
  // detections: [
  //   { x: 100, y: 100, width: 200, height: 300, confidence: 0.95, class: 'dress' },
  //   { x: 350, y: 150, width: 100, height: 150, confidence: 0.87, class: 'shoe' },
  // ]

  // Step 3: Crop each detection and embed it
  for (const detection of detections) {
    // Crop the image region
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = detection.width;
    croppedCanvas.height = detection.height;
    const ctx = croppedCanvas.getContext('2d');
    ctx.drawImage(
      image,
      detection.x,
      detection.y,
      detection.width,
      detection.height,
      0,
      0,
      detection.width,
      detection.height
    );
    const croppedBlob = await new Promise<Blob>((resolve) => {
      croppedCanvas.toBlob(resolve);
    });

    // Embed the cropped region
    const embeddingResult = await visionCore.embed(croppedBlob);
    console.log(`${detection.class} embedding:`, embeddingResult.embedding);
  }
}
```

### 8.3 Face Recognition / Verification

**Goal**: Compare two face photos. Are they the same person?

**Setup**:

```typescript
import { VisionCore } from '@vision-core/core';
import { createOnnxEngine } from '@vision-core/engine-onnx';
import { WebImageAdapter } from '@vision-core/adapter-web';
import * as ort from 'onnxruntime-web';
import { l2Normalize } from '@vision-core/types';

// Use ArcFace or FaceNet ONNX model
const visionCore = new VisionCore(engine, adapter);

await visionCore.initialize({
  modelSource: './models/arcface_resnet100.onnx',
  modelLoader: async (source) => {
    const response = await fetch(source);
    return response.arrayBuffer();
  },
  inputTensorName: 'input',
  outputTensorName: 'output',
  inputWidth: 112,    // ArcFace expects 112×112
  inputHeight: 112,
  channels: 3,
  channelOrder: 'CHW',
  // ArcFace normalization (specific to this model)
  normalization: { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] },
});

// Helper: Calculate cosine similarity
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let similarity = 0;
  for (let i = 0; i < a.length; i++) {
    similarity += a[i] * b[i];
  }
  return similarity;
}

// Compare two faces
async function areSamePerson(
  faceImage1: File,
  faceImage2: File,
  threshold: number = 0.6  // tunable: higher = stricter matching
): Promise<{ same: boolean; similarity: number }> {
  const result1 = await visionCore.embed(faceImage1);
  const result2 = await visionCore.embed(faceImage2);

  // ArcFace models often normalize internally, but it's safe to do it again
  const embedding1 = l2Normalize(result1.embedding);
  const embedding2 = l2Normalize(result2.embedding);

  const similarity = cosineSimilarity(embedding1, embedding2);

  return {
    same: similarity >= threshold,
    similarity,
  };
}

// Usage:
const file1Input = document.getElementById('face1') as HTMLInputElement;
const file2Input = document.getElementById('face2') as HTMLInputElement;
const compareBtn = document.getElementById('compare');

compareBtn.addEventListener('click', async () => {
  const file1 = file1Input.files?.[0];
  const file2 = file2Input.files?.[0];

  if (file1 && file2) {
    const { same, similarity } = await areSamePerson(file1, file2);
    console.log(`Same person: ${same}`);
    console.log(`Similarity score: ${(similarity * 100).toFixed(2)}%`);
  }
});
```

### 8.4 Visual Search Engine (Server-Side)

**Goal**: Client uploads a photo. Server searches a database of 1 million product embeddings.

**Server side**:

```typescript
// Store embeddings in a vector database (e.g., Pinecone, Milvus, pgvector)
import Pinecone from '@pinecone-database/pinecone';

const client = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = client.Index('products');

// Upsert (add or update) product embeddings
async function indexProducts(products: Array<{ id: string; embedding: number[] }>) {
  await index.upsert({
    vectors: products.map(({ id, embedding }) => ({
      id,
      values: embedding,
      metadata: { productId: id },
    })),
  });
}

// Search for similar products
async function searchProducts(
  queryEmbedding: number[],
  topK: number = 10
): Promise<Array<{ id: string; score: number }>> {
  const results = await index.query({
    vector: queryEmbedding,
    topK,
    includeMetadata: true,
  });

  return results.matches.map((match) => ({
    id: match.metadata.productId as string,
    score: match.score,
  }));
}

// Express endpoint
app.post('/search', async (req, res) => {
  const { embedding } = req.body;  // sent from client

  const results = await searchProducts(Array.from(embedding), 10);
  res.json({ similarProducts: results });
});
```

**Client side**:

```typescript
// User uploads image
const fileInput = document.getElementById('imageInput') as HTMLInputElement;

fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  // Embed locally
  const result = await visionCore.embed(file);
  const embedding = l2Normalize(result.embedding);

  // Send to server for search
  const response = await fetch('/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding: Array.from(embedding) }),
  });

  const { similarProducts } = await response.json();
  console.log('Top results:', similarProducts);
});
```

### 8.5 Image Classification with Reference Embeddings

**Goal**: Classify images into 5 categories without training a model. Just use embedding similarity.

**Setup**:

```typescript
// Step 1: Create reference embeddings for each category
const categories = ['shoes', 'dresses', 'hats', 'bags', 'gloves'];

interface CategoryReference {
  name: string;
  embedding: Float32Array;
}

const references: CategoryReference[] = [];

async function createReferences() {
  // For each category, embed a few representative images
  for (const category of categories) {
    // You might have 3-5 reference images per category
    const referenceImages = [
      `/reference-images/${category}-1.jpg`,
      `/reference-images/${category}-2.jpg`,
      `/reference-images/${category}-3.jpg`,
    ];

    // Average their embeddings
    const embeddings: Float32Array[] = [];
    for (const url of referenceImages) {
      const result = await visionCore.embed(url);
      embeddings.push(l2Normalize(result.embedding));
    }

    const avgEmbedding = new Float32Array(embeddings[0].length);
    for (let i = 0; i < avgEmbedding.length; i++) {
      avgEmbedding[i] = embeddings.reduce((sum, e) => sum + e[i], 0) / embeddings.length;
    }

    references.push({
      name: category,
      embedding: l2Normalize(avgEmbedding),  // normalize the average
    });
  }
}

// Step 2: Classify new image
async function classifyImage(imageFile: File): Promise<{ category: string; confidence: number }> {
  const result = await visionCore.embed(imageFile);
  const embedding = l2Normalize(result.embedding);

  // Find most similar category
  let bestMatch = { category: '', similarity: -2 };

  for (const ref of references) {
    let similarity = 0;
    for (let i = 0; i < embedding.length; i++) {
      similarity += embedding[i] * ref.embedding[i];
    }

    if (similarity > bestMatch.similarity) {
      bestMatch = { category: ref.name, similarity };
    }
  }

  return { category: bestMatch.category, confidence: Math.max(0, bestMatch.similarity) };
}

// Usage:
const imageInput = document.getElementById('imageInput') as HTMLInputElement;
imageInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    const { category, confidence } = await classifyImage(file);
    console.log(`Predicted: ${category} (${(confidence * 100).toFixed(1)}% confident)`);
  }
});
```

---

## How to Use Your Own Trained Model

### Step 1: Train Your Model

Here's a minimal PyTorch example that trains a simple feature extractor:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# Load pre-trained ResNet-50, remove the final classification layer
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  # Remove classifier, keep features

# Or, create a custom architecture:
class CustomFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ... more layers
        self.fc = nn.Linear(2048, 512)  # Output 512-dimensional embeddings

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # ... more layers
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CustomFeatureExtractor()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.TripletMarginLoss()  # Or nn.CosineSimilarityLoss, etc.

# Dummy data (replace with your dataset)
class YourDataset(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # Return (anchor_image, positive_image, negative_image) for triplet loss
        # Or just (image, label) if using classification loss
        pass

train_loader = DataLoader(YourDataset(), batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(10):
    for batch in train_loader:
        # Your training logic
        pass

model.eval()
```

### Step 2: Export to ONNX

```python
import torch
import torch.onnx

# Load your trained model
model = CustomFeatureExtractor()
model.load_state_dict(torch.load('my-model.pth'))
model.eval()

# Create a dummy input matching your model's expected shape
# (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'my-model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=12,  # Use 12+ for better compatibility
    do_constant_folding=True,
    export_params=True,
    verbose=False,
)

print('Model exported to my-model.onnx')
```

### Step 3: Verify Tensor Names and Shapes with Netron

1. Go to https://netron.app/
2. Upload `my-model.onnx`
3. Find the input node (leftmost):
   - Name: `input` (or whatever you named it)
   - Shape: should be `[1, 3, 224, 224]` or similar
4. Find the output node (rightmost):
   - Name: `output` (or whatever you named it)
   - Shape: should be `[1, 512]` for 512-dimensional embeddings

### Step 4: Determine Normalization Values

If you used ImageNet normalization during training, use the standard values:

```python
# ImageNet normalization (from torchvision.transforms)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

If you used custom normalization, calculate from your training data:

```python
import numpy as np
from torchvision import datasets

# Compute mean and std from your training images
dataset = datasets.ImageFolder('path/to/train', transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=32, shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)
total = 0

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total += batch_samples

mean /= total
std /= total

print(f'mean = {mean.tolist()}')
print(f'std = {std.tolist()}')
```

### Step 5: Create ModelConfig

```typescript
import { ModelConfig } from '@vision-core/types';

const myModelConfig: ModelConfig = {
  modelSource: './models/my-model.onnx',
  modelLoader: async (source: string) => {
    const response = await fetch(source);
    return response.arrayBuffer();
  },
  inputTensorName: 'input',      // from Netron
  outputTensorName: 'output',    // from Netron
  inputWidth: 224,               // from your training
  inputHeight: 224,
  channels: 3,
  channelOrder: 'CHW',           // PyTorch uses CHW
  normalization: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
};

const visionCore = new VisionCore(engine, adapter);
await visionCore.initialize(myModelConfig);

const embedding = await visionCore.embed(imageFile);
console.log('Embedding:', embedding.embedding);
```

---

## Common Models and Their Configs

| Model | Size | inputWidth | inputHeight | channelOrder | mean | std | Notes |
|-------|------|-----------|------------|--------------|------|-----|-------|
| MobileNetV3 | 224×224 | 224 | 224 | CHW | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | Fast, good for mobile |
| ResNet-50 | 224×224 | 224 | 224 | CHW | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | Standard, balanced |
| EfficientNet-B0 | 224×224 | 224 | 224 | CHW | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | Efficient family |
| EfficientNet-B7 | 600×600 | 600 | 600 | CHW | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | Larger, more accurate |
| CLIP ViT-B/32 | 224×224 | 224 | 224 | CHW | [0.48145466, 0.4578275, 0.40821073] | [0.26862954, 0.26130258, 0.27577711] | Multimodal, vision+text |
| ArcFace-ResNet100 | 112×112 | 112 | 112 | CHW | [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] | Face recognition |
| DINOv2 | 224×224 | 224 | 224 | CHW | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | Self-supervised, strong |
| YOLOv8n (detection) | 640×640 | 640 | 640 | CHW | [0, 0, 0] | [1, 1, 1] | Object detection, not embeddings |

---

## Error Handling

### EngineNotInitializedError

**When**: You try to embed before calling `initialize()`.

```typescript
const visionCore = new VisionCore(engine, adapter);
const embedding = await visionCore.embed(image);
// ❌ Error: Engine has not been initialized. Call initialize() first.
```

**Fix**: Call `initialize()` first.

```typescript
await visionCore.initialize(modelConfig);
const embedding = await visionCore.embed(image);  // ✓ Works
```

### InvalidInputError

**When**: The image input is invalid (e.g., wrong format, corrupted file).

**Fix**: Ensure your input is valid:
- File/Blob: use files from file input or fetch
- URL string: ensure CORS is enabled
- HTMLImageElement: ensure it's loaded (image.complete === true)
- ImageBitmap: ensure it was created successfully

### InferenceError

**When**: The model fails to run (usually due to wrong ModelConfig).

**Common causes**:

1. **Wrong tensor name**: `InferenceError: Output tensor 'wrong_output' not found`
   - Fix: Use Netron to verify the correct tensor name

2. **Wrong input shape**: Model expects [1, 3, 224, 224] but got [1, 3, 640, 640]
   - Fix: Ensure inputWidth and inputHeight match the model

3. **Wrong channelOrder**: You passed HWC but model expects CHW
   - Fix: Check Netron for the correct order

4. **Wrong normalization**: Pixel values way outside expected range
   - Fix: Verify mean/std values are correct

### AdapterError

**When**: The ImageAdapter fails to preprocess (web) or the PixelDecoder fails (React Native).

**Web causes**:
- Canvas context failed
- Image failed to load
- Invalid blob/file

**React Native causes**:
- PixelDecoder threw an error
- Image library failed to decode
- File path doesn't exist

**Fix**: Log the error cause:

```typescript
try {
  await visionCore.embed(image);
} catch (error) {
  if (error instanceof AdapterError) {
    console.error('Adapter failed:', error.cause);
  }
}
```

---

## Tips and Troubleshooting

### Model Outputs Garbage Embeddings

**Symptom**: Embeddings look like random numbers, or two different images produce nearly identical embeddings.

**Debugging steps**:

1. **Check normalization values**:
   ```typescript
   // ImageNet standard (most common)
   normalization: {
     mean: [0.485, 0.456, 0.406],
     std: [0.229, 0.224, 0.225],
   }
   ```

2. **Verify channelOrder**:
   - Use Netron to check input shape
   - `[1, 3, 224, 224]` → CHW
   - `[1, 224, 224, 3]` → HWC

3. **Test with a known image**:
   ```typescript
   // Use a standard test image and verify output isn't all zeros/ones
   const testUrl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png';
   const result = await visionCore.embed(testUrl);
   console.log('Min:', Math.min(...result.embedding));
   console.log('Max:', Math.max(...result.embedding));
   // Should be in reasonable range, not all zeros
   ```

4. **Check tensor shape at runtime**:
   - The adapter should produce `{ data: Float32Array(150528), shape: [3, 224, 224] }` for CHW
   - Or `{ data: Float32Array(150528), shape: [224, 224, 3] }` for HWC

### "Output tensor not found" Error

**Fix**: Use Netron to find the exact output tensor name:
1. Upload your `.onnx` to netron.app
2. Click on the rightmost node
3. Look at the "Name" field in the properties panel
4. Use that exact name in `outputTensorName`

### Performance Tips

**Web**:
```typescript
// Use WebGL backend for better performance
import * as ort from 'onnxruntime-web';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
ort.env.webgl.disablePack = false;  // Enable WebGL optimizations
```

**React Native**:
- Use bilinear resize if possible (avoid larger images)
- Cache embeddings if doing many comparisons

**General**:
- Resize images before embedding (don't pass 4K images)
- Batch multiple embeddings if possible
- Use a faster model (MobileNetV3) for real-time use cases
- Always call `dispose()` when done to free memory

### Memory Management

**Always clean up**:

```typescript
// Bad: memory leak
const visionCore = new VisionCore(engine, adapter);
await visionCore.initialize(config);
// ... use it ...
// forget to dispose

// Good: clean memory
const visionCore = new VisionCore(engine, adapter);
try {
  await visionCore.initialize(config);
  // ... use it ...
} finally {
  await visionCore.dispose();  // Always clean up
}

// Even better: use a cleanup function
async function withVisionCore<T>(
  fn: (vc: VisionCore) => Promise<T>
): Promise<T> {
  const visionCore = new VisionCore(engine, adapter);
  try {
    await visionCore.initialize(config);
    return await fn(visionCore);
  } finally {
    await visionCore.dispose();
  }
}

// Usage:
const result = await withVisionCore(async (vc) => {
  return await vc.embed(image);
});
```

### Handling Large Databases

For similarity search with millions of embeddings, use a vector database:

- **Pinecone**: Cloud, easiest to set up
- **Milvus**: Open-source, self-hosted
- **pgvector**: PostgreSQL extension
- **Weaviate**: Open-source, rich filtering
- **Qdrant**: Modern, fast

Store embeddings as `Float32Array → Array.from()` → JSON, then query with your embedding.

---

## Summary

You now have a complete understanding of Vision-Core:

- **What it does**: Converts images to embeddings (fingerprints)
- **How it works**: Image → Preprocess (resize, normalize) → Inference (run model) → Embedding
- **How to use it**: Create VisionCore, initialize with ModelConfig, embed images
- **Real-world applications**: Image search, classification, face recognition, quality control
- **How to train your own**: PyTorch → ONNX → ModelConfig → Vision-Core

The library handles the complexity of different platforms (web, React Native), different models (YOLO, CLIP, ArcFace, custom), and different image inputs (URLs, files, base64). You just provide the model and image, and get back meaningful numerical representations ready for comparison, clustering, or further processing.

Good luck with your vision projects!
