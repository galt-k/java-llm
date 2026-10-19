package com.galt.llm;

import com.galt.llm.config.GPTConfig;

import java.util.random.RandomGenerator;

public class GPTModel {
    private final GPTConfig gptConfig;
    private final Tensor tokenEmb;
    private final Tensor posEmb;
    private final TransformerBlock[] transformerBlocks;
    private final LayerNorm finalLayerNorm;
    private final Tensor outHead;

    public GPTModel(
            GPTConfig config,
            Tensor tokEmb,
            Tensor posEmb,
            TransformerBlock[] blocks,
            LayerNorm finalNorm,
            Tensor outHead
    ) {
        this.gptConfig = config;
        this.tokenEmb = tokEmb;
        this.posEmb = posEmb;
        this.transformerBlocks = blocks;
        this.finalLayerNorm = finalNorm;
        this.outHead = outHead;
    }

    public Tensor forward(int[][] inIdx) {
        int B = inIdx.length;
        int T = inIdx[0].length;
        int D = gptConfig.embDim();

        // 1. Embedding lookup and position addition
        Tensor x = Tensor.zeros(B, T, D);
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int tokenID = inIdx[b][t];
                for (int d = 0; d < D; d++) {
                    float tokVal = tokenEmb.get(tokenID, d);
                    float posVal = posEmb.get(t, d);
                    x.set(tokVal + posVal, b, t, d);
                }
            }
        }

        // 2. Pass through all Transformer blocks
        for (TransformerBlock block : transformerBlocks) {
            x = block.forward(x, true);
        }

        // 3. Final LayerNorm
        x = finalLayerNorm.forward(x);

        // 4. Vocabulary projection -> [B, T, vocabSize]
        return x.matmul(outHead);
    }

    public int[][] generateTextSimple(int[][] idx, int maxNewTokens, int contextSize) {
        int B = idx.length;
        int[][] currIdx = new int[B][];
        for (int b = 0; b < B; b++) {
            currIdx[b] = idx[b].clone();
        }

        for (int step = 0; step < maxNewTokens; step++) {
            int currentLength = currIdx[0].length;

            // 1. Crop context if needed (take last contextSize tokens)
            int startIndex = Math.max(0, currentLength - contextSize);
            int condLength = currentLength - startIndex;
            int[][] idxCond = new int[B][condLength];
            for (int b = 0; b < B; b++) {
                System.arraycopy(currIdx[b], startIndex, idxCond[b], 0, condLength);
            }

            // 2. Run forward pass
            Tensor logits = this.forward(idxCond); // shape [B, condLength, vocabSize]
            int lastTime = condLength - 1;

            // 3. Greedy argmax: pick the token with highest score for each batch
            int[][] nextIdx = new int[B][currentLength + 1];
            for (int b = 0; b < B; b++) {
                int bestToken = 0;
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < gptConfig.vocabSize(); v++) {
                    float score = logits.get(b, lastTime, v);
                    if (score > maxVal) {
                        maxVal = score;
                        bestToken = v;
                    }
                }

                // 4. Append best token to sequence
                System.arraycopy(currIdx[b], 0, nextIdx[b], 0, currentLength);
                nextIdx[b][currentLength] = bestToken;
            }

            currIdx = nextIdx;
        }

        return currIdx;
    }

    /**
     * Autoregressive text generation with Temperature scaling, Top-k filtering,
     * and multinomial sampling (or deterministic greedy argmax if temperature == 0.0).
     *
     * @param idx          initial prompt token IDs of shape [B, T]
     * @param maxNewTokens maximum number of new tokens to generate
     * @param contextSize  maximum context window size (crops from the left if exceeded)
     * @param temperature  scaling factor for randomness (0.0 = deterministic greedy argmax)
     * @param topK         optional integer to restrict sampling to the top-k highest logits (null = disabled)
     * @param eosId        optional end-of-sequence token ID to terminate generation early (null = disabled)
     * @param rng          random number generator for inverse-CDF multinomial sampling (can be null if temperature == 0.0)
     * @return 2D integer array of generated tokens containing the original prompt plus new tokens
     */
    public int[][] generate(
            int[][] idx,
            int maxNewTokens,
            int contextSize,
            float temperature,
            Integer topK,
            Integer eosId,
            RandomGenerator rng
    ) {
        int B = idx.length;
        int[][] currIdx = new int[B][];
        for (int b = 0; b < B; b++) {
            currIdx[b] = idx[b].clone();
        }

        for (int step = 0; step < maxNewTokens; step++) {
            int currentLength = currIdx[0].length;

            // 1. Crop context to contextSize if current sequence length exceeds it
            int startIndex = Math.max(0, currentLength - contextSize);
            int condLength = currentLength - startIndex;
            int[][] idxCond = new int[B][condLength];
            for (int b = 0; b < B; b++) {
                System.arraycopy(currIdx[b], startIndex, idxCond[b], 0, condLength);
            }

            // 2. Forward pass to get unnormalized vocabulary logits [B, condLength, vocabSize]
            Tensor logits = this.forward(idxCond);
            int lastTime = condLength - 1;

            // 3. For each batch item, extract last-step logits, apply Top-k, and sample next token
            int[][] nextIdx = new int[B][currentLength + 1];
            boolean allFinished = (eosId != null);

            for (int b = 0; b < B; b++) {
                // Extract 1D logits for the last token position
                float[] lastLogits = new float[gptConfig.vocabSize()];
                for (int v = 0; v < gptConfig.vocabSize(); v++) {
                    lastLogits[v] = logits.get(b, lastTime, v);
                }

                // Apply Top-k filtering if specified (overwrites non-top-k logits with -Infinity)
                if (topK != null) {
                    SamplingUtils.applyTopK(lastLogits, topK);
                }

                // Sample next token (handles temperature scaling, softmax, and inverse-CDF sampling)
                int nextToken = SamplingUtils.sample(lastLogits, temperature, rng);

                // Copy existing sequence and append newly generated token
                System.arraycopy(currIdx[b], 0, nextIdx[b], 0, currentLength);
                nextIdx[b][currentLength] = nextToken;

                // Check for early stopping condition
                if (eosId != null && nextToken != eosId) {
                    allFinished = false;
                }
            }

            currIdx = nextIdx;

            // If an EOS token was specified and all batch sequences generated it, terminate early
            if (allFinished) {
                break;
            }
        }

        return currIdx;
    }
}
