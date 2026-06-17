import { gather, Rank, Tensor } from "@tensorflow/tfjs";

/**
 * Nested-array indexing: each argument is an index on dim 0, then dim 1, etc.
 *
 *   batch[0]       →  tensorGet(batch, 0)       // [4, 256]
 *   batch[0][0]    →  tensorGet(batch, 0, 0)    // [256]
 *   batch[0][0][5] →  tensorGet(batch, 0, 0, 5) // scalar
 *
 * Uses gather on axis 0 at each step (same as peeling nested arrays).
 */
export function tensorGet(tensor: Tensor<Rank>, indices: number[]) {
  if (indices.length === 0) {
    throw new Error("tensorGet: indices array cannot be empty");
  }

  if (indices.length > tensor.shape.length) {
    throw new Error(
      `tensorGet: expected at most ${tensor.shape.length} indices, got ${indices.length}`
    );
  }

  let result = tensor;

  for (const index of indices) {
    const axisSize = result.shape[0];

    if (index < 0 || index >= axisSize) {
      throw new Error(
        `tensorGet: index ${index} out of bounds for axis 0 (size ${axisSize})`
      );
    }

    result = gather(result, index, 0);
  }

  return result
}
