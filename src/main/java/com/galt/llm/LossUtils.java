package com.galt.llm;

public class LossUtils {

    /**
     * Calculates the average Cross-Entropy Loss across all tokens in the batch
     * using the numerically stable Log-Sum-Exp trick.
     *
     * @param logits  Tensor of shape [B, T, V]
     * @param targets 2D array of shape [B, T] with ground-truth token IDs
     * @return scalar average cross-entropy loss
     */
    public static float crossEntropyLoss(Tensor logits, int[][] targets) {
        int[] shape = logits.shape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("Expected 3D logits [B, T, V], got rank " + shape.length);
        }

        int B = shape[0];
        int T = shape[1];
        int V = shape[2];

        if (targets.length != B) {
            throw new IllegalArgumentException("Expected targets batch size " + B + ", got " + targets.length);
        }

        float totalLoss = 0.0f;

        // Iterate over each batch and each token position
        for (int b = 0; b < B; b++) {
            if (targets[b].length != T) {
                throw new IllegalArgumentException("Expected sequence length " + T + " for batch " + b + ", got " + targets[b].length);
            }
            for (int t = 0; t < T; t++) {
                int targetId = targets[b][t];

                // Pass 1: Find the max logit across vocabulary (prevents exp overflow)
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < V; v++) {
                    maxLogit = Math.max(maxLogit, logits.get(b, t, v));
                }

                // Pass 2: Sum the exponentiated shifted logits
                float sumExp = 0.0f;
                for (int v = 0; v < V; v++) {
                    sumExp += (float) Math.exp(logits.get(b, t, v) - maxLogit);
                }

                // Compute this token's loss using Log-Sum-Exp:
                // tokenLoss = -logits[targetId] + maxLogit + ln(sumExp)
                float tokenLoss = -logits.get(b, t, targetId) + maxLogit + (float) Math.log(sumExp);
                totalLoss += tokenLoss;
            }
        }

        // Return the average loss across all B * T tokens
        return totalLoss / (B * T);
    }

    /**
     * Calculates perplexity from cross-entropy loss: exp(loss).
     */
    public static float perplexity(float loss) {
        return (float) Math.exp(loss);
    }
}
