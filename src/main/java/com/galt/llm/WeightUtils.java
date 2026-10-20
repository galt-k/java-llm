package com.galt.llm;

import java.util.Arrays;

public class WeightUtils {

    /**
     * Splits a 2D tensor [rows, totalCols] along the column axis into numSplits equal tensors [rows, splitCols].
     * Used for separating fused OpenAI c_attn weights [768, 2304] into W_query, W_key, W_value [768, 768].
     */
    public static Tensor[] splitColumns(Tensor fused, int numSplits) {
        if (fused.rank() != 2 || fused.shape()[1] % numSplits != 0) {
            throw new IllegalArgumentException(
                "Expected 2D tensor with columns divisible by " + numSplits + ", got shape " + Arrays.toString(fused.shape())
            );
        }

        int rows = fused.shape()[0];
        int totalCols = fused.shape()[1];
        int splitCols = totalCols / numSplits;

        Tensor[] splits = new Tensor[numSplits];
        for (int k = 0; k < numSplits; k++) {
            Tensor chunk = Tensor.zeros(rows, splitCols);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < splitCols; c++) {
                    chunk.set(fused.get(r, k * splitCols + c), r, c);
                }
            }
            splits[k] = chunk;
        }

        return splits;
    }

    /**
     * Splits a 1D tensor [totalDim] into numSplits equal 1D vectors [splitDim].
     * Used for separating fused OpenAI c_attn bias [2304] into b_query, b_key, b_value [768].
     */
    public static Tensor[] splitVector(Tensor fused, int numSplits) {
        if (fused.rank() != 1 || fused.shape()[0] % numSplits != 0) {
            throw new IllegalArgumentException(
                "Expected 1D tensor with length divisible by " + numSplits + ", got shape " + Arrays.toString(fused.shape())
            );
        }

        int totalDim = fused.shape()[0];
        int splitDim = totalDim / numSplits;

        Tensor[] splits = new Tensor[numSplits];
        for (int k = 0; k < numSplits; k++) {
            Tensor chunk = Tensor.zeros(splitDim);
            for (int i = 0; i < splitDim; i++) {
                chunk.set(fused.get(k * splitDim + i), i);
            }
            splits[k] = chunk;
        }

        return splits;
    }

    /**
     * Enforces shape invariant when loading/assigning weights, preventing sideways matrix loading.
     */
    public static void assertShape(Tensor expected, Tensor loaded, String layerName) {
        if (!Arrays.equals(expected.shape(), loaded.shape())) {
            throw new IllegalArgumentException(
                "Shape mismatch for layer '" + layerName + "': expected " +
                Arrays.toString(expected.shape()) + ", but loaded " + Arrays.toString(loaded.shape())
            );
        }
    }
}
