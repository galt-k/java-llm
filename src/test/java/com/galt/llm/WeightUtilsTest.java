package com.galt.llm;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WeightUtilsTest {

    @Test
    @DisplayName("splitColumns correctly splits [rows, totalCols] into numSplits equal [rows, splitCols] tensors")
    void testSplitColumns() {
        // [2 rows, 6 cols] -> split into 3 chunks of [2 rows, 2 cols]
        float[] data = {
            1.0f, 2.0f,  3.0f, 4.0f,  5.0f, 6.0f,
            7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f
        };
        Tensor fused = Tensor.of(data, 2, 6);

        Tensor[] splits = WeightUtils.splitColumns(fused, 3);
        assertEquals(3, splits.length);

        // Chunk 0 (Q): cols 0, 1
        assertArrayEquals(new int[]{2, 2}, splits[0].shape());
        assertEquals(1.0f, splits[0].get(0, 0));
        assertEquals(2.0f, splits[0].get(0, 1));
        assertEquals(7.0f, splits[0].get(1, 0));
        assertEquals(8.0f, splits[0].get(1, 1));

        // Chunk 1 (K): cols 2, 3
        assertArrayEquals(new int[]{2, 2}, splits[1].shape());
        assertEquals(3.0f, splits[1].get(0, 0));
        assertEquals(4.0f, splits[1].get(0, 1));
        assertEquals(9.0f, splits[1].get(1, 0));
        assertEquals(10.0f, splits[1].get(1, 1));

        // Chunk 2 (V): cols 4, 5
        assertArrayEquals(new int[]{2, 2}, splits[2].shape());
        assertEquals(5.0f, splits[2].get(0, 0));
        assertEquals(6.0f, splits[2].get(0, 1));
        assertEquals(11.0f, splits[2].get(1, 0));
        assertEquals(12.0f, splits[2].get(1, 1));
    }

    @Test
    @DisplayName("splitVector correctly splits [totalDim] into numSplits equal [splitDim] vectors")
    void testSplitVector() {
        // [6] -> split into 3 chunks of [2]
        float[] data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
        Tensor fused = Tensor.of(data, 6);

        Tensor[] splits = WeightUtils.splitVector(fused, 3);
        assertEquals(3, splits.length);

        assertArrayEquals(new int[]{2}, splits[0].shape());
        assertEquals(10.0f, splits[0].get(0));
        assertEquals(20.0f, splits[0].get(1));

        assertArrayEquals(new int[]{2}, splits[1].shape());
        assertEquals(30.0f, splits[1].get(0));
        assertEquals(40.0f, splits[1].get(1));

        assertArrayEquals(new int[]{2}, splits[2].shape());
        assertEquals(50.0f, splits[2].get(0));
        assertEquals(60.0f, splits[2].get(1));
    }

    @Test
    @DisplayName("assertShape validates shape matching and throws on mismatch")
    void testAssertShape() {
        Tensor a = Tensor.zeros(768, 768);
        Tensor b = Tensor.zeros(768, 768);
        Tensor c = Tensor.zeros(768, 2304);

        assertDoesNotThrow(() -> WeightUtils.assertShape(a, b, "testLayer"));
        assertThrows(IllegalArgumentException.class, () -> WeightUtils.assertShape(a, c, "mismatchedLayer"));
    }

    @Test
    @DisplayName("MultiHeadAttention correctly computes with QKV and OutProj biases")
    void testAttentionWithBiases() {
        int embDim = 4;
        int numHeads = 2;

        float[] wData = new float[embDim * embDim];
        java.util.Arrays.fill(wData, 0.1f);
        Tensor wQ = Tensor.of(wData.clone(), embDim, embDim);
        Tensor wK = Tensor.of(wData.clone(), embDim, embDim);
        Tensor wV = Tensor.of(wData.clone(), embDim, embDim);
        Tensor outProj = Tensor.of(wData.clone(), embDim, embDim);

        float[] bData = new float[embDim];
        java.util.Arrays.fill(bData, 0.05f);
        Tensor bQ = Tensor.of(bData.clone(), embDim);
        Tensor bK = Tensor.of(bData.clone(), embDim);
        Tensor bV = Tensor.of(bData.clone(), embDim);
        Tensor bOut = Tensor.of(bData.clone(), embDim);

        MultiHeadAttention mhaWithBias = new MultiHeadAttention(wQ, bQ, wK, bK, wV, bV, outProj, bOut, numHeads);
        MultiHeadAttention mhaWithoutBias = new MultiHeadAttention(wQ, wK, wV, outProj, numHeads);

        Tensor x = Tensor.zeros(1, 3, embDim); // B=1, T=3, D=4
        Tensor outWith = mhaWithBias.forward(x, true);
        Tensor outWithout = mhaWithoutBias.forward(x, true);

        assertArrayEquals(new int[]{1, 3, embDim}, outWith.shape());
        // Since input is all zeros, output without bias is all zeros, but output with bias is non-zero
        assertNotEquals(outWith.get(0, 0, 0), outWithout.get(0, 0, 0));
        assertEquals(0.0f, outWithout.get(0, 0, 0), 1e-6f);
    }
}
