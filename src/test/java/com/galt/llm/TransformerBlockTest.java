package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TransformerBlockTest {

    @Test
    void testTensorAddition() {
        Tensor a = Tensor.of(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
        Tensor b = Tensor.of(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, 2, 2);

        Tensor c = a.add(b);
        assertArrayEquals(new int[]{2, 2}, c.shape());
        assertEquals(11.0f, c.get(0, 0));
        assertEquals(22.0f, c.get(0, 1));
        assertEquals(33.0f, c.get(1, 0));
        assertEquals(44.0f, c.get(1, 1));

        // Mismatched shapes must throw IllegalArgumentException
        Tensor diffShape = Tensor.of(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4, 1);
        assertThrows(IllegalArgumentException.class, () -> a.add(diffShape));
    }

    @Test
    void testTransformerBlockForwardPass() {
        int B = 2;
        int T = 3;
        int embDim = 4;
        int numHeads = 2;
        int hiddenDim = 4 * embDim; // 16

        // 1. Attention components
        float[] wData = new float[embDim * embDim];
        java.util.Arrays.fill(wData, 0.1f);
        Tensor wQ = Tensor.of(wData, embDim, embDim);
        Tensor wK = Tensor.of(wData, embDim, embDim);
        Tensor wV = Tensor.of(wData, embDim, embDim);
        Tensor outProj = Tensor.of(wData, embDim, embDim);
        MultiHeadAttention att = new MultiHeadAttention(wQ, wK, wV, outProj, numHeads);

        // 2. FeedForward components
        float[] w1Data = new float[embDim * hiddenDim];
        java.util.Arrays.fill(w1Data, 0.05f);
        Tensor w1 = Tensor.of(w1Data, embDim, hiddenDim);

        float[] b1Data = new float[hiddenDim];
        java.util.Arrays.fill(b1Data, 0.1f);
        Tensor b1 = Tensor.of(b1Data, hiddenDim);

        float[] w2Data = new float[hiddenDim * embDim];
        java.util.Arrays.fill(w2Data, 0.05f);
        Tensor w2 = Tensor.of(w2Data, hiddenDim, embDim);

        float[] b2Data = new float[embDim];
        java.util.Arrays.fill(b2Data, 0.1f);
        Tensor b2 = Tensor.of(b2Data, embDim);
        FeedForward ff = new FeedForward(w1, b1, w2, b2);

        // 3. LayerNorms
        LayerNorm norm1 = new LayerNorm(embDim);
        LayerNorm norm2 = new LayerNorm(embDim);

        // 4. Assemble TransformerBlock
        TransformerBlock block = new TransformerBlock(att, ff, norm1, norm2);

        // Input tensor: [B=2, T=3, embDim=4]
        float[] inData = new float[B * T * embDim];
        for (int i = 0; i < inData.length; i++) {
            inData[i] = (i + 1) * 0.2f;
        }
        Tensor x = Tensor.of(inData, B, T, embDim);

        // Run forward pass with causal mask
        Tensor out = block.forward(x, true);

        // Verify shape is strictly preserved
        assertArrayEquals(new int[]{B, T, embDim}, out.shape());

        // Verify all outputs are valid finite numbers (no NaNs or Infs)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < embDim; d++) {
                    assertFalse(Float.isNaN(out.get(b, t, d)), "Output must not contain NaN");
                    assertFalse(Float.isInfinite(out.get(b, t, d)), "Output must not contain Infinity");
                }
            }
        }

        // Verify that residual connection actually modified the output from the raw input
        boolean hasChanged = false;
        for (int i = 0; i < inData.length; i++) {
            if (Math.abs(out.data()[i] - inData[i]) > 1e-4f) {
                hasChanged = true;
                break;
            }
        }
        assertTrue(hasChanged, "TransformerBlock must refine the input features");
    }
}
