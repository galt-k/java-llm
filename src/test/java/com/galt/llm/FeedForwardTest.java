package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FeedForwardTest {

    @Test
    void testGELUMathematicalApproximation() {
        GELU gelu = new GELU();

        // Test tensor with critical values
        float[] inputs = {0.0f, 1.0f, -2.0f, 2.0f};
        Tensor x = Tensor.of(inputs, 4);

        Tensor out = gelu.forward(x);

        // GELU(0) = 0.0
        assertEquals(0.0f, out.get(0), 1e-5f);

        // GELU(1) ≈ 0.84119 (exact PyTorch tanh-approximation value)
        assertEquals(0.84119f, out.get(1), 1e-4f);

        // GELU(-2) ≈ -0.0454f
        assertEquals(-0.0454f, out.get(2), 1e-3f);

        // GELU(2) ≈ 1.9545f
        assertEquals(1.9545f, out.get(3), 1e-3f);
    }

    @Test
    void testAddBiasBroadcast() {
        // 2D tensor: [2, 3]
        Tensor x = Tensor.of(new float[]{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }, 2, 3);

        Tensor bias = Tensor.of(new float[]{10.0f, 20.0f, 30.0f}, 3);

        Tensor out = x.addBias(bias);
        assertArrayEquals(new int[]{2, 3}, out.shape());

        // Row 0: [11, 22, 33]
        assertEquals(11.0f, out.get(0, 0));
        assertEquals(22.0f, out.get(0, 1));
        assertEquals(33.0f, out.get(0, 2));

        // Row 1: [14, 25, 36]
        assertEquals(14.0f, out.get(1, 0));
        assertEquals(25.0f, out.get(1, 1));
        assertEquals(36.0f, out.get(1, 2));
    }

    @Test
    void testFeedForwardExpansionAndContraction() {
        int B = 2;
        int T = 3;
        int embDim = 4;
        int hiddenDim = 4 * embDim; // 16

        // Inputs: [2, 3, 4]
        float[] inputData = new float[B * T * embDim];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (i + 1) * 0.1f;
        }
        Tensor x = Tensor.of(inputData, B, T, embDim);

        // Weights & Biases
        // W1: [4, 16], B1: [16]
        float[] w1Data = new float[embDim * hiddenDim];
        java.util.Arrays.fill(w1Data, 0.05f);
        Tensor w1 = Tensor.of(w1Data, embDim, hiddenDim);

        float[] b1Data = new float[hiddenDim];
        java.util.Arrays.fill(b1Data, 0.1f);
        Tensor b1 = Tensor.of(b1Data, hiddenDim);

        // W2: [16, 4], B2: [4]
        float[] w2Data = new float[hiddenDim * embDim];
        java.util.Arrays.fill(w2Data, 0.05f);
        Tensor w2 = Tensor.of(w2Data, hiddenDim, embDim);

        float[] b2Data = new float[embDim];
        java.util.Arrays.fill(b2Data, 0.2f);
        Tensor b2 = Tensor.of(b2Data, embDim);

        FeedForward ff = new FeedForward(w1, b1, w2, b2);

        Tensor out = ff.forward(x);

        // Verify shape is preserved: [B, T, embDim] -> [2, 3, 4]
        assertArrayEquals(new int[]{B, T, embDim}, out.shape());

        // Verify all outputs are valid numbers (no NaNs or Infs)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embDim; d++) {
                    assertFalse(Float.isNaN(out.get(b, t, d)));
                    assertFalse(Float.isInfinite(out.get(b, t, d)));
                    assertTrue(out.get(b, t, d) > 0.0f);
                }
            }
        }
    }
}
