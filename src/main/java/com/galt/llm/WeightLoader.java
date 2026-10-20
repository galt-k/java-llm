package com.galt.llm;

import com.galt.llm.config.GPTConfig;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

public class WeightLoader {

    /**
     * Loads a raw Little-Endian 32-bit float binary file directly into a strided Tensor.
     */
    public static Tensor loadTensor(Path path, int... shape) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) channel.size())
                    .order(ByteOrder.LITTLE_ENDIAN);
            channel.read(buffer);
            buffer.flip();

            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            float[] data = new float[floatBuffer.remaining()];
            floatBuffer.get(data);

            return Tensor.of(data, shape);
        }
    }

    /**
     * Assembles a full GPTModel from a dictionary of pretrained weights.
     * Maps OpenAI / HuggingFace layer names into our modular Transformer architecture:
     * - Splits fused c_attn [768, 2304] weights into W_q, W_k, W_v [768, 768]
     * - Splits fused c_attn [2304] bias into b_q, b_k, b_v [768]
     * - Reuses wte [50257, 768] transposed as outHead [768, 50257] (weight tying)
     * - Assembles all N transformer blocks and final LayerNorm
     */
    public static GPTModel buildGPTModel(GPTConfig config, Map<String, Tensor> weights) {
        int embDim = config.embDim();
        int vocabSize = config.vocabSize();
        int contextLength = config.contextLength();
        int nLayers = config.nLayers();
        int nHeads = config.nHeads();

        // 1. Token Embeddings and Positional Embeddings
        Tensor wte = weights.get("wte");
        WeightUtils.assertShape(Tensor.zeros(vocabSize, embDim), wte, "wte");

        Tensor wpe = weights.get("wpe");
        WeightUtils.assertShape(Tensor.zeros(contextLength, embDim), wpe, "wpe");

        // 2. Weight Tying: outHead is wte transposed [embDim, vocabSize]
        Tensor outHead = wte.transpose(0, 1).contiguous();

        // 3. Assemble all Transformer blocks
        TransformerBlock[] blocks = new TransformerBlock[nLayers];

        for (int l = 0; l < nLayers; l++) {
            String prefix = "blocks." + l + ".";

            // LayerNorm 1
            Tensor ln1Weight = weights.get(prefix + "ln_1.weight");
            Tensor ln1Bias = weights.get(prefix + "ln_1.bias");
            WeightUtils.assertShape(Tensor.zeros(embDim), ln1Weight, prefix + "ln_1.weight");
            WeightUtils.assertShape(Tensor.zeros(embDim), ln1Bias, prefix + "ln_1.bias");
            LayerNorm ln1 = new LayerNorm(ln1Weight, ln1Bias);

            // MultiHeadAttention: Split fused c_attn weights and biases
            Tensor fusedAttnWeight = weights.get(prefix + "attn.c_attn.weight");
            Tensor fusedAttnBias = weights.get(prefix + "attn.c_attn.bias");
            WeightUtils.assertShape(Tensor.zeros(embDim, 3 * embDim), fusedAttnWeight, prefix + "attn.c_attn.weight");
            WeightUtils.assertShape(Tensor.zeros(3 * embDim), fusedAttnBias, prefix + "attn.c_attn.bias");

            Tensor[] qkvWeights = WeightUtils.splitColumns(fusedAttnWeight, 3);
            Tensor[] qkvBiases = WeightUtils.splitVector(fusedAttnBias, 3);

            Tensor wQuery = qkvWeights[0];
            Tensor bQuery = qkvBiases[0];
            Tensor wKey = qkvWeights[1];
            Tensor bKey = qkvBiases[1];
            Tensor wValue = qkvWeights[2];
            Tensor bValue = qkvBiases[2];

            // Attention Output Projection
            Tensor outProjWeight = weights.get(prefix + "attn.c_proj.weight");
            Tensor outProjBias = weights.get(prefix + "attn.c_proj.bias");
            WeightUtils.assertShape(Tensor.zeros(embDim, embDim), outProjWeight, prefix + "attn.c_proj.weight");
            WeightUtils.assertShape(Tensor.zeros(embDim), outProjBias, prefix + "attn.c_proj.bias");

            MultiHeadAttention att = new MultiHeadAttention(
                    wQuery, bQuery,
                    wKey, bKey,
                    wValue, bValue,
                    outProjWeight, outProjBias,
                    nHeads
            );

            // LayerNorm 2
            Tensor ln2Weight = weights.get(prefix + "ln_2.weight");
            Tensor ln2Bias = weights.get(prefix + "ln_2.bias");
            WeightUtils.assertShape(Tensor.zeros(embDim), ln2Weight, prefix + "ln_2.weight");
            WeightUtils.assertShape(Tensor.zeros(embDim), ln2Bias, prefix + "ln_2.bias");
            LayerNorm ln2 = new LayerNorm(ln2Weight, ln2Bias);

            // FeedForward (MLP)
            int hiddenDim = 4 * embDim;
            Tensor fcWeight = weights.get(prefix + "mlp.c_fc.weight");
            Tensor fcBias = weights.get(prefix + "mlp.c_fc.bias");
            WeightUtils.assertShape(Tensor.zeros(embDim, hiddenDim), fcWeight, prefix + "mlp.c_fc.weight");
            WeightUtils.assertShape(Tensor.zeros(hiddenDim), fcBias, prefix + "mlp.c_fc.bias");

            Tensor projWeight = weights.get(prefix + "mlp.c_proj.weight");
            Tensor projBias = weights.get(prefix + "mlp.c_proj.bias");
            WeightUtils.assertShape(Tensor.zeros(hiddenDim, embDim), projWeight, prefix + "mlp.c_proj.weight");
            WeightUtils.assertShape(Tensor.zeros(embDim), projBias, prefix + "mlp.c_proj.bias");

            FeedForward ff = new FeedForward(fcWeight, fcBias, projWeight, projBias);

            // Transformer Block
            blocks[l] = new TransformerBlock(att, ff, ln1, ln2);
        }

        // 4. Final LayerNorm
        Tensor lnFWeight = weights.get("ln_f.weight");
        Tensor lnFBias = weights.get("ln_f.bias");
        WeightUtils.assertShape(Tensor.zeros(embDim), lnFWeight, "ln_f.weight");
        WeightUtils.assertShape(Tensor.zeros(embDim), lnFBias, "ln_f.bias");
        LayerNorm finalNorm = new LayerNorm(lnFWeight, lnFBias);

        return new GPTModel(config, wte, wpe, blocks, finalNorm, outHead);
    }
}
