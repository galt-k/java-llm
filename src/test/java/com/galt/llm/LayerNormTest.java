package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LayerNormTest {

    @Test
    void testLayerNorm2DNormalization() {
        int T = 3;
        int embDim = 4;
        // Sample 2D input with arbitrary drift
        float[] data = {
            10.0f,  20.0f,  30.0f,  40.0f,  // Row 0
            -5.0f,  -2.0f,   1.0f,   4.0f,  // Row 1
             0.1f,   0.2f,   0.3f,   0.4f   // Row 2
        };

        Tensor x = Tensor.of(data, T, embDim);
        LayerNorm ln = new LayerNorm(embDim);

        Tensor out = ln.forward(x);
        assertArrayEquals(new int[]{T, embDim}, out.shape());

        // For each token row, post-normalization mean must be ~0.0f and variance ~1.0f (within eps tolerance)
        for (int t = 0; t < T; t++) {
            float mean = 0.0f;
            for (int i = 0; i < embDim; i++) {
                mean += out.get(t, i);
            }
            mean /= embDim;
            assertEquals(0.0f, mean, 1e-5f, "Row " + t + " mean must be ~0.0f");

            float variance = 0.0f;
            for (int i = 0; i < embDim; i++) {
                float diff = out.get(t, i) - mean;
                variance += diff * diff;
            }
            variance /= embDim;
            assertEquals(1.0f, variance, 1e-3f, "Row " + t + " variance must be ~1.0f");
        }
    }

    @Test
    void testLayerNorm3DWithScaleAndShift() {
        int B = 2;
        int T = 2;
        int embDim = 3;

        float[] data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f
        };
        Tensor x = Tensor.of(data, B, T, embDim);

        // Custom scale and shift
        Tensor scale = Tensor.of(new float[]{2.0f, 2.0f, 2.0f}, embDim);
        Tensor shift = Tensor.of(new float[]{0.5f, 0.5f, 0.5f}, embDim);
        LayerNorm ln = new LayerNorm(scale, shift);

        Tensor out = ln.forward(x);
        assertArrayEquals(new int[]{B, T, embDim}, out.shape());

        // For each batch and token row:
        // With scale = 2.0 and shift = 0.5, mean should be ~0.5f, variance should be ~4.0f (scale^2)
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float mean = 0.0f;
                for (int i = 0; i < embDim; i++) {
                    mean += out.get(b, t, i);
                }
                mean /= embDim;
                assertEquals(0.5f, mean, 1e-5f);

                float variance = 0.0f;
                for (int i = 0; i < embDim; i++) {
                    float diff = out.get(b, t, i) - mean;
                    variance += diff * diff;
                }
                variance /= embDim;
                assertEquals(4.0f, variance, 1e-3f);
            }
        }
    }
}
