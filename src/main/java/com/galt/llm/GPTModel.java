package com.galt.llm;

import com.galt.llm.config.GPTConfig;

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
}
