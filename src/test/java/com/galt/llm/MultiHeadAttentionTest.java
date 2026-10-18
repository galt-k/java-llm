package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MultiHeadAttentionTest {

    private static final float[] INPUTS_FLAT = {
        0.43f, 0.15f, 0.89f, // Your     (x^1)
        0.55f, 0.87f, 0.66f, // journey  (x^2)
        0.57f, 0.85f, 0.64f, // starts   (x^3)
        0.22f, 0.58f, 0.33f, // with     (x^4)
        0.77f, 0.25f, 0.10f, // one      (x^5)
        0.05f, 0.80f, 0.55f  // step     (x^6)
    };

    private static final float[] W_QUERY_SEED_123 = {
        0.2961f, 0.5166f,
        0.2517f, 0.6886f,
        0.0740f, 0.8665f
    };

    private static final float[] W_KEY_SEED_123 = {
        0.1366f, 0.1025f,
        0.1841f, 0.7264f,
        0.3153f, 0.6871f
    };

    private static final float[] W_VALUE_SEED_123 = {
        0.0756f, 0.1966f,
        0.3164f, 0.4017f,
        0.1186f, 0.8274f
    };

    // Identity projection for out_proj (shape [2, 2])
    private static final float[] OUT_PROJ_IDENTITY = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    @Test
    void testDivisibilityValidation() {
        Tensor wQ = Tensor.of(W_QUERY_SEED_123, 3, 2);
        Tensor wK = Tensor.of(W_KEY_SEED_123, 3, 2);
        Tensor wV = Tensor.of(W_VALUE_SEED_123, 3, 2);
        Tensor outProj = Tensor.of(OUT_PROJ_IDENTITY, 2, 2);

        // dOut = 2 is not divisible by numHeads = 3
        assertThrows(IllegalArgumentException.class, () ->
            new MultiHeadAttention(wQ, wK, wV, outProj, 3)
        );
    }

    @Test
    void testBatchedMultiHeadAttentionShapeAndInvariants() {
        int B = 2;
        int T = 6;
        int dIn = 3;
        int dOut = 2;
        int numHeads = 2;

        // Duplicate the 6 tokens into 2 identical batches: [2, 6, 3]
        float[] batchData = new float[B * T * dIn];
        System.arraycopy(INPUTS_FLAT, 0, batchData, 0, INPUTS_FLAT.length);
        System.arraycopy(INPUTS_FLAT, 0, batchData, INPUTS_FLAT.length, INPUTS_FLAT.length);

        Tensor x = Tensor.of(batchData, B, T, dIn);
        Tensor wQ = Tensor.of(W_QUERY_SEED_123, dIn, dOut);
        Tensor wK = Tensor.of(W_KEY_SEED_123, dIn, dOut);
        Tensor wV = Tensor.of(W_VALUE_SEED_123, dIn, dOut);
        Tensor outProj = Tensor.of(OUT_PROJ_IDENTITY, dOut, dOut);

        MultiHeadAttention mha = new MultiHeadAttention(wQ, wK, wV, outProj, numHeads);

        // Run forward pass with causal masking enabled
        Tensor output = mha.forward(x, true);

        // 1. Output shape must be [B, T, dOut] -> [2, 6, 2]
        assertArrayEquals(new int[]{B, T, dOut}, output.shape());

        // 2. Numerical validity: No NaNs or Infs
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < dOut; d++) {
                    assertFalse(Float.isNaN(output.get(b, t, d)), "Output must not contain NaN");
                    assertFalse(Float.isInfinite(output.get(b, t, d)), "Output must not contain Infinity");
                }
            }
        }

        // 3. Batch consistency: Since Batch 0 and Batch 1 had identical inputs, their outputs must match exactly
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < dOut; d++) {
                assertEquals(
                    output.get(0, t, d),
                    output.get(1, t, d),
                    1e-6f,
                    "Batch 0 and Batch 1 outputs must match"
                );
            }
        }
    }
}
