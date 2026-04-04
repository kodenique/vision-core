export function l2Normalize(embedding: Float32Array): Float32Array {
  let sumOfSquares = 0;
  for (let i = 0; i < embedding.length; i++) {
    sumOfSquares += embedding[i] * embedding[i];
  }
  const norm = Math.sqrt(sumOfSquares);
  if (norm === 0) {
    return new Float32Array(embedding.length);
  }
  const result = new Float32Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) {
    result[i] = embedding[i] / norm;
  }
  return result;
}
