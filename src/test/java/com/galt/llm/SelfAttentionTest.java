package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SelfAttentionTest {

    // Inputs: "Your journey starts with one step" (shape [6, 3])
    private static final float[] INPUTS_FLAT = {
        0.43f, 0.15f, 0.89f, // Your     (x^1)
        0.55f, 0.87f, 0.66f, // journey  (x^2)
        0.57f, 0.85f, 0.64f, // starts   (x^3)
        0.22f, 0.58f, 0.33f, // with     (x^4)
        0.77f, 0.25f, 0.10f, // one      (x^5)
        0.05f, 0.80f, 0.55f  // step     (x^6)
    };

    // PyTorch torch.manual_seed(123) weights (shape [3, 2])
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

    // Expected Chapter 3.4 Unmasked Outputs (shape [6, 2])
    private static final float[][] EXPECTED_SA_V1 = {
        {0.2996f, 0.8053f},
        {0.3061f, 0.8210f},
        {0.3058f, 0.8203f},
        {0.2948f, 0.7939f},
        {0.2927f, 0.7891f},
        {0.2990f, 0.8040f}
    };

    @Test
    void testUnmaskedSelfAttentionAgainstPyTorchGroundTruth() {
        Tensor x = Tensor.of(INPUTS_FLAT, 6, 3);
        Tensor wQ = Tensor.of(W_QUERY_SEED_123, 3, 2);
        Tensor wK = Tensor.of(W_KEY_SEED_123, 3, 2);
        Tensor wV = Tensor.of(W_VALUE_SEED_123, 3, 2);

        SelfAttention sa = new SelfAttention(wQ, wK, wV);

        // Run forward without causal mask (causalMask = false)
        Tensor context = sa.forward(x, false);

        assertArrayEquals(new int[]{6, 2}, context.shape());

        // Verify full matrix against PyTorch Listing 3.1 / 3.4 reference values (tolerance 1e-4)
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 2; c++) {
                assertEquals(
                    EXPECTED_SA_V1[r][c],
                    context.get(r, c),
                    1e-4f,
                    "Mismatch at context[" + r + ", " + c + "]"
                );
            }
        }
    }

    @Test
    void testCausalMaskedSelfAttention() {
        Tensor x = Tensor.of(INPUTS_FLAT, 6, 3);
        Tensor wQ = Tensor.of(W_QUERY_SEED_123, 3, 2);
        Tensor wK = Tensor.of(W_KEY_SEED_123, 3, 2);
        Tensor wV = Tensor.of(W_VALUE_SEED_123, 3, 2);

        SelfAttention sa = new SelfAttention(wQ, wK, wV);

        // Run forward WITH causal mask
        Tensor context = sa.forward(x, true);
        assertArrayEquals(new int[]{6, 2}, context.shape());

        // Token 0 can only attend to itself, so context[0] must equal V[0]!
        Tensor V = x.matmul(wV);
        assertEquals(V.get(0, 0), context.get(0, 0), 1e-4f);
        assertEquals(V.get(0, 1), context.get(0, 1), 1e-4f);

        // Ensure no NaNs or Infs anywhere in the causal output
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 2; c++) {
                assertFalse(Float.isNaN(context.get(r, c)));
                assertFalse(Float.isInfinite(context.get(r, c)));
            }
        }
    }
}
