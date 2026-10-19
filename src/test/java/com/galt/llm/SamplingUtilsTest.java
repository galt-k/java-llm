package com.galt.llm;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class SamplingUtilsTest {

    // 9-token toy vocabulary from Section 5.3.1:
    // 0:"closer", 1:"every", 2:"effort", 3:"forward", 4:"inches", 5:"moves", 6:"pizza", 7:"toward", 8:"you"
    private static final float[] NEXT_TOKEN_LOGITS = {
            4.51f, 0.89f, -1.90f, 6.75f, 1.63f, -1.62f, -1.89f, 6.28f, 1.79f
    };

    @Test
    @DisplayName("Section 5.3.1: Greedy decoding (temperature = 0.0) selects highest logit index 3 ('forward')")
    void testGreedyDecodingSection531() {
        float[] logits = NEXT_TOKEN_LOGITS.clone();

        int selected = SamplingUtils.sample(logits, 0.0f, null);
        assertEquals(3, selected, "Greedy decoding must select index 3 ('forward') with score 6.75");
    }

    @Test
    @DisplayName("Section 5.3.2: Top-K filtering (k = 3) preserves top 3 scores and masks others to -Infinity")
    void testTopKFilteringSection532() {
        float[] logits = NEXT_TOKEN_LOGITS.clone();

        SamplingUtils.applyTopK(logits, 3);

        // Top 3 scores in NEXT_TOKEN_LOGITS are:
        // index 3: 6.75f
        // index 7: 6.28f
        // index 0: 4.51f
        assertEquals(4.51f, logits[0], 1e-5f);
        assertEquals(6.75f, logits[3], 1e-5f);
        assertEquals(6.28f, logits[7], 1e-5f);

        // All other 6 indices must be Float.NEGATIVE_INFINITY
        assertEquals(Float.NEGATIVE_INFINITY, logits[1]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[2]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[4]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[5]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[6]);
        assertEquals(Float.NEGATIVE_INFINITY, logits[8]);
    }

    @Test
    @DisplayName("Top-K sampling strictly forbids masked tokens from being sampled")
    void testSamplingNeverPicksMaskedTokens() {
        float[] logits = NEXT_TOKEN_LOGITS.clone();
        SamplingUtils.applyTopK(logits, 3);

        Set<Integer> allowedTokens = Set.of(0, 3, 7);
        Random rng = new Random(123);

        // Draw 200 random samples; every sample MUST be in the top-3 allowed set
        for (int i = 0; i < 200; i++) {
            // clone because temperature scaling mutates logits in-place
            float[] stepLogits = logits.clone();
            int token = SamplingUtils.sample(stepLogits, 1.0f, rng);
            assertTrue(allowedTokens.contains(token),
                    "Sampled token " + token + " was not in allowed Top-K set " + allowedTokens);
        }
    }

    @Test
    @DisplayName("Low temperature sharpens distribution toward the highest score")
    void testLowTemperatureSharpness() {
        float[] logits = NEXT_TOKEN_LOGITS.clone();
        Random rng = new Random(42);

        // At temperature = 0.05, the peak at index 3 (score 6.75) dominates completely
        int countTop = 0;
        for (int i = 0; i < 100; i++) {
            float[] stepLogits = logits.clone();
            int token = SamplingUtils.sample(stepLogits, 0.05f, rng);
            if (token == 3) {
                countTop++;
            }
        }
        assertTrue(countTop >= 99, "At temperature 0.05, top token should be sampled >= 99% of the time");
    }

    @Test
    @DisplayName("Negative temperature throws IllegalArgumentException")
    void testNegativeTemperature() {
        float[] logits = NEXT_TOKEN_LOGITS.clone();
        assertThrows(IllegalArgumentException.class, () ->
                SamplingUtils.sample(logits, -0.5f, null)
        );
    }
}
