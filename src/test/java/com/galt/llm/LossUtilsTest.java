package com.galt.llm;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LossUtilsTest {

    @Test
    @DisplayName("Uniform logits across vocabulary V should yield loss = ln(V) and perplexity = V")
    void testUniformLogitsMatchesTheoreticalEntropy() {
        int B = 2;
        int T = 3;
        int V = 50257;

        // All logits are 0.0f (uniform distribution across vocabulary)
        Tensor logits = Tensor.zeros(B, T, V);
        int[][] targets = {
            {10, 500, 50256},
            {0, 25000, 100}
        };

        float loss = LossUtils.crossEntropyLoss(logits, targets);
        float expectedLoss = (float) Math.log(V); // ln(50257) ≈ 10.8249
        assertEquals(expectedLoss, loss, 1e-4f, "Uniform logits loss must equal ln(V)");

        float perplexity = LossUtils.perplexity(loss);
        assertEquals((float) V, perplexity, 1.0f, "Uniform perplexity must equal vocabulary size V");
    }

    @Test
    @DisplayName("Confidently correct predictions should yield near-zero loss and perplexity ≈ 1.0")
    void testConfidentlyCorrectPredictions() {
        int B = 1;
        int T = 2;
        int V = 10;

        Tensor logits = Tensor.zeros(B, T, V);
        int[][] targets = {{3, 7}};

        // Assign high positive logits to the target tokens
        logits.set(50.0f, 0, 0, 3);
        logits.set(50.0f, 0, 1, 7);

        float loss = LossUtils.crossEntropyLoss(logits, targets);
        assertTrue(loss >= 0.0f, "Loss must be non-negative");
        assertTrue(loss < 1e-5f, "Loss for confidently correct predictions must approach 0.0");

        float perplexity = LossUtils.perplexity(loss);
        assertEquals(1.0f, perplexity, 1e-3f, "Perplexity for perfect predictions must approach 1.0");
    }

    @Test
    @DisplayName("Log-Sum-Exp trick prevents IEEE 754 overflow on extreme logits")
    void testLogSumExpStabilityExtremeLogits() {
        int B = 1;
        int T = 1;
        int V = 4;

        // Logits in the thousands would overflow Math.exp(1000.0f) to +Infinity without max subtraction
        float[] extremeData = {1000.0f, 1005.0f, 995.0f, 1002.0f};
        Tensor logits = Tensor.of(extremeData, B, T, V);
        int[][] targets = {{1}}; // target is index 1 (logit 1005.0f)

        float loss = LossUtils.crossEntropyLoss(logits, targets);
        assertFalse(Float.isNaN(loss), "Loss must not be NaN");
        assertFalse(Float.isInfinite(loss), "Loss must not be Infinite");
        assertTrue(loss >= 0.0f, "Loss must be non-negative");

        float perplexity = LossUtils.perplexity(loss);
        assertFalse(Float.isNaN(perplexity), "Perplexity must not be NaN");
        assertFalse(Float.isInfinite(perplexity), "Perplexity must not be Infinite");
    }

    @Test
    @DisplayName("Section 5.1.2 ground-truth perplexity check from spec")
    void testSpecSection512Fixture() {
        float expectedLoss = 10.7940f;
        float expectedPerplexity = 48725.82f;

        float actualPerplexity = LossUtils.perplexity(expectedLoss);
        assertEquals(expectedPerplexity, actualPerplexity, 5.0f,
                "Perplexity should match Chapter 5 Section 5.1.2 within float32 tolerance");
    }

    @Test
    @DisplayName("Validation checks for mismatched shapes")
    void testShapeValidation() {
        Tensor logits = Tensor.zeros(2, 3, 10);
        int[][] wrongBatch = {{1, 2, 3}};
        assertThrows(IllegalArgumentException.class, () -> LossUtils.crossEntropyLoss(logits, wrongBatch));

        int[][] wrongSeq = {{1, 2}, {1, 2}};
        assertThrows(IllegalArgumentException.class, () -> LossUtils.crossEntropyLoss(logits, wrongSeq));

        Tensor wrongRank = Tensor.zeros(2, 3);
        int[][] targets = {{1, 2, 3}, {1, 2, 3}};
        assertThrows(IllegalArgumentException.class, () -> LossUtils.crossEntropyLoss(wrongRank, targets));
    }
}
