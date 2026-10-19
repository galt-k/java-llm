package com.galt.llm;

import java.util.PriorityQueue;
import java.util.random.RandomGenerator;

public class SamplingUtils {

    /**
     * Filters logits in-place to keep only the top-k highest values.
     * All logits below the k-th highest are set to Float.NEGATIVE_INFINITY.
     */
    public static void applyTopK(float[] logits, int k) {
        if (k >= logits.length || k <= 0) {
            // No filtering needed.
            return;
        }

        // Maintain a min-heap of size k to find the k-th largest element
        PriorityQueue<Float> minHeap = new PriorityQueue<>(k);
        for (float v : logits) {
            if (minHeap.size() < k) {
                minHeap.add(v);
            } else if (v > minHeap.peek()) {
                minHeap.poll();
                minHeap.add(v);
            }
        }
        float threshold = minHeap.peek();

        // Overwrite any logit below the threshold with -Infinity
        for (int index = 0; index < logits.length; index++) {
            if (logits[index] < threshold) {
                logits[index] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    /**
     * Samples the next token index from logits using temperature scaling and multinomial sampling.
     * If temperature is 0.0f, performs deterministic greedy argmax.
     *
     * @param logits      1D array of logits for a single sequence position (length V)
     * @param temperature temperature scaling factor (>= 0.0f)
     * @param rng         random number generator for inverse-CDF sampling
     * @return chosen token index in 0..V-1
     */
    public static int sample(float[] logits, float temperature, RandomGenerator rng) {
        if (temperature < 0.0f) {
            throw new IllegalArgumentException("Temperature must be non-negative, got " + temperature);
        }

        // Branch 1: Deterministic Greedy (argmax)
        if (temperature == 0.0f) {
            return findMaxIndex(logits);
        }

        // Branch 2: Stochastic Sampling
        // Step 2a: Temperature Scaling
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] != Float.NEGATIVE_INFINITY) {
                logits[i] /= temperature;
            }
        }

        // Step 2b: Numerically stable softmax
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float val : logits) {
            if (val > maxLogit) {
                maxLogit = val;
            }
        }

        float sumExp = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] == Float.NEGATIVE_INFINITY) {
                probs[i] = 0.0f;
            } else {
                probs[i] = (float) Math.exp(logits[i] - maxLogit);
                sumExp += probs[i];
            }
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }

        // Step 2c: Inverse-CDF Multinomial Sampling
        float u = rng.nextFloat();
        float cdf = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cdf += probs[i];
            if (u < cdf) {
                return i;
            }
        }

        return probs.length - 1; // Fallback for float rounding
    }

    /**
     * Finds the index of the maximum element in the logits array (argmax).
     * Linear scan is optimal O(V) with zero object allocation and JIT vectorization.
     */
    public static int findMaxIndex(float[] logits) {
        int bestIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
