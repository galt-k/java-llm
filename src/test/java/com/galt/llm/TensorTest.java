package com.galt.llm;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    @Test
    void testContiguousStridesAndIndexing() {
        float[] data = new float[24];
        for (int i = 0; i < 24; i++) {
            data[i] = i;
        }

        Tensor t = Tensor.of(data, 2, 3, 4);
        assertArrayEquals(new int[]{12, 4, 1}, t.strides());
        assertTrue(t.isContiguous());

        assertEquals(0.0f, t.get(0, 0, 0));
        assertEquals(12.0f, t.get(1, 0, 0));
        assertEquals(23.0f, t.get(1, 2, 3));
    }

    @Test
    void testTransposeZeroCopyAndValues() {
        // 2x2 matrix:
        // [ 1.0, 2.0 ]
        // [ 3.0, 4.0 ]
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor t = Tensor.of(data, 2, 2);

        Tensor transposed = t.transpose(0, 1);

        // Metadata check: zero-copy shares the same data buffer
        assertSame(t.data(), transposed.data(), "Transpose must not copy the data buffer");
        assertArrayEquals(new int[]{2, 2}, transposed.shape());
        assertArrayEquals(new int[]{1, 2}, transposed.strides());
        assertFalse(transposed.isContiguous());

        // Value check: reading transposed matrix
        // [ 1.0, 3.0 ]
        // [ 2.0, 4.0 ]
        assertEquals(1.0f, transposed.get(0, 0));
        assertEquals(3.0f, transposed.get(0, 1));
        assertEquals(2.0f, transposed.get(1, 0));
        assertEquals(4.0f, transposed.get(1, 1));
    }

    @Test
    void testContiguityAndView() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        Tensor t = Tensor.of(data, 2, 3);
        assertTrue(t.isContiguous());

        // Reshaping a contiguous tensor works zero-copy
        Tensor reshaped = t.view(3, 2);
        assertSame(t.data(), reshaped.data());
        assertArrayEquals(new int[]{3, 2}, reshaped.shape());

        // Transposing breaks contiguity
        Tensor transposed = t.transpose(0, 1); // shape [3, 2], strides [1, 3]
        assertFalse(transposed.isContiguous());

        // Calling view on non-contiguous tensor must fail
        assertThrows(IllegalStateException.class, () -> transposed.view(6));

        // Making it contiguous creates a new buffer in row-major order
        Tensor madeContiguous = transposed.contiguous();
        assertTrue(madeContiguous.isContiguous());
        assertNotSame(transposed.data(), madeContiguous.data());

        // Now view succeeds
        Tensor flat = madeContiguous.view(6);
        assertArrayEquals(new int[]{6}, flat.shape());
        // Expected order: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        float[] expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
        for (int i = 0; i < 6; i++) {
            assertEquals(expected[i], flat.get(i), 1e-6f);
        }
    }

    @Test
    void testMatmul2D() {
        // [2, 3] @ [3, 2] -> [2, 2]
        // A = [[1, 2, 3], [4, 5, 6]]
        // B = [[7, 8], [9, 1], [2, 3]]
        Tensor a = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor b = Tensor.of(new float[]{7, 8, 9, 1, 2, 3}, 3, 2);

        Tensor c = a.matmul(b);
        assertArrayEquals(new int[]{2, 2}, c.shape());

        // Row 0: [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [7+18+6, 8+2+9] = [31, 19]
        // Row 1: [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [28+45+12, 32+5+18] = [85, 55]
        assertEquals(31.0f, c.get(0, 0), 1e-6f);
        assertEquals(19.0f, c.get(0, 1), 1e-6f);
        assertEquals(85.0f, c.get(1, 0), 1e-6f);
        assertEquals(55.0f, c.get(1, 1), 1e-6f);
    }

    @Test
    void testSoftmaxRowSumsAndNumericalStability() {
        // Regular softmax test
        Tensor t = Tensor.of(new float[]{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }, 2, 3);

        Tensor sm = t.softmax(-1);
        assertArrayEquals(new int[]{2, 3}, sm.shape());

        // Each row must sum to 1.0f
        float row0Sum = sm.get(0, 0) + sm.get(0, 1) + sm.get(0, 2);
        float row1Sum = sm.get(1, 0) + sm.get(1, 1) + sm.get(1, 2);
        assertEquals(1.0f, row0Sum, 1e-6f);
        assertEquals(1.0f, row1Sum, 1e-6f);

        // Extreme values test (proving max-subtraction prevents overflow)
        Tensor extreme = Tensor.of(new float[]{
            1000.0f, 1001.0f, 999.0f
        }, 1, 3);

        Tensor extremeSm = extreme.softmax(-1);
        for (int i = 0; i < 3; i++) {
            assertFalse(Float.isNaN(extremeSm.get(0, i)), "Softmax must not produce NaN on large inputs");
            assertTrue(extremeSm.get(0, i) >= 0.0f && extremeSm.get(0, i) <= 1.0f);
        }
        float extremeSum = extremeSm.get(0, 0) + extremeSm.get(0, 1) + extremeSm.get(0, 2);
        assertEquals(1.0f, extremeSum, 1e-5f);
    }
}
