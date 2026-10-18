package com.galt.llm;

import com.galt.llm.config.GPTConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GPTModelTest {

    private GPTModel createTestModel(GPTConfig config) {
        int vocabSize = config.vocabSize();
        int contextLength = config.contextLength();
        int embDim = config.embDim();
        int nHeads = config.nHeads();
        int nLayers = config.nLayers();
        int hiddenDim = 4 * embDim;

        // 1. Token and Positional Embeddings
        float[] tokData = new float[vocabSize * embDim];
        for (int i = 0; i < tokData.length; i++) {
            tokData[i] = (i % 10) * 0.05f;
        }
        Tensor tokEmb = Tensor.of(tokData, vocabSize, embDim);

        float[] posData = new float[contextLength * embDim];
        for (int i = 0; i < posData.length; i++) {
            posData[i] = (i % 8) * 0.02f;
        }
        Tensor posEmb = Tensor.of(posData, contextLength, embDim);

        // 2. Transformer Blocks
        TransformerBlock[] blocks = new TransformerBlock[nLayers];
        for (int l = 0; l < nLayers; l++) {
            // Attention weights
            float[] wData = new float[embDim * embDim];
            java.util.Arrays.fill(wData, 0.05f);
            Tensor wQ = Tensor.of(wData, embDim, embDim);
            Tensor wK = Tensor.of(wData, embDim, embDim);
            Tensor wV = Tensor.of(wData, embDim, embDim);
            Tensor outProj = Tensor.of(wData, embDim, embDim);
            MultiHeadAttention att = new MultiHeadAttention(wQ, wK, wV, outProj, nHeads);

            // FeedForward weights
            float[] w1Data = new float[embDim * hiddenDim];
            java.util.Arrays.fill(w1Data, 0.02f);
            Tensor w1 = Tensor.of(w1Data, embDim, hiddenDim);

            float[] b1Data = new float[hiddenDim];
            java.util.Arrays.fill(b1Data, 0.01f);
            Tensor b1 = Tensor.of(b1Data, hiddenDim);

            float[] w2Data = new float[hiddenDim * embDim];
            java.util.Arrays.fill(w2Data, 0.02f);
            Tensor w2 = Tensor.of(w2Data, hiddenDim, embDim);

            float[] b2Data = new float[embDim];
            java.util.Arrays.fill(b2Data, 0.01f);
            Tensor b2 = Tensor.of(b2Data, embDim);
            FeedForward ff = new FeedForward(w1, b1, w2, b2);

            LayerNorm norm1 = new LayerNorm(embDim);
            LayerNorm norm2 = new LayerNorm(embDim);

            blocks[l] = new TransformerBlock(att, ff, norm1, norm2);
        }

        // 3. Final LayerNorm & Output Head
        LayerNorm finalNorm = new LayerNorm(embDim);
        float[] outHeadData = new float[embDim * vocabSize];
        for (int i = 0; i < outHeadData.length; i++) {
            outHeadData[i] = (i % 7) * 0.03f;
        }
        Tensor outHead = Tensor.of(outHeadData, embDim, vocabSize);

        return new GPTModel(config, tokEmb, posEmb, blocks, finalNorm, outHead);
    }

    @Test
    void testGPTModelForwardPass() {
        GPTConfig config = new GPTConfig(10, 8, 4, 2, 2, false);
        GPTModel model = createTestModel(config);

        // Input: Batch=2, Tokens=3
        int[][] inIdx = {
            {1, 4, 2},
            {0, 3, 5}
        };

        Tensor logits = model.forward(inIdx);

        // Verify shape is [B, T, vocabSize] -> [2, 3, 10]
        assertArrayEquals(new int[]{2, 3, 10}, logits.shape());

        // Verify all scores are finite (no NaNs or Infs)
        for (int b = 0; b < 2; b++) {
            for (int t = 0; t < 3; t++) {
                for (int v = 0; v < 10; v++) {
                    assertFalse(Float.isNaN(logits.get(b, t, v)), "Logits must not contain NaN");
                    assertFalse(Float.isInfinite(logits.get(b, t, v)), "Logits must not contain Infinity");
                }
            }
        }
    }

    @Test
    void testAutoregressiveGeneration() {
        GPTConfig config = new GPTConfig(10, 8, 4, 2, 2, false);
        GPTModel model = createTestModel(config);

        // Prompt of 2 tokens: [1, 2]
        int[][] prompt = {
            {1, 2}
        };

        int maxNewTokens = 3;
        int contextSize = 8;

        // Generate 3 new tokens
        int[][] generated = model.generateTextSimple(prompt, maxNewTokens, contextSize);

        // Total sequence length must be original 2 + 3 = 5 tokens
        assertEquals(1, generated.length);
        assertEquals(5, generated[0].length);

        // The first 2 tokens must remain the original prompt
        assertEquals(1, generated[0][0]);
        assertEquals(2, generated[0][1]);

        // The 3 newly generated tokens must be valid vocabulary IDs (0 <= token < 10)
        for (int i = 2; i < 5; i++) {
            int token = generated[0][i];
            assertTrue(token >= 0 && token < 10, "Generated token must be a valid vocab ID: " + token);
        }
    }
}
