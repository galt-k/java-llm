package com.galt.llm;

import com.galt.llm.config.GPTConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class WeightLoaderTest {

    @Test
    @DisplayName("loadTensor reads raw little-endian float32 binary file accurately into Tensor")
    void testLoadTensorBinaryFile(@TempDir Path tempDir) throws IOException {
        Path tempFile = tempDir.resolve("test_tensor.bin");

        float[] expected = {1.5f, -2.25f, 3.125f, 0.0f, 100.5f, -99.75f};
        try (FileChannel channel = FileChannel.open(tempFile, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            ByteBuffer buf = ByteBuffer.allocateDirect(expected.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : expected) {
                buf.putFloat(v);
            }
            buf.flip();
            channel.write(buf);
        }

        Tensor loaded = WeightLoader.loadTensor(tempFile, 2, 3);
        assertArrayEquals(new int[]{2, 3}, loaded.shape());

        assertEquals(1.5f, loaded.get(0, 0), 1e-6f);
        assertEquals(-2.25f, loaded.get(0, 1), 1e-6f);
        assertEquals(3.125f, loaded.get(0, 2), 1e-6f);
        assertEquals(0.0f, loaded.get(1, 0), 1e-6f);
        assertEquals(100.5f, loaded.get(1, 1), 1e-6f);
        assertEquals(-99.75f, loaded.get(1, 2), 1e-6f);
    }

    @Test
    @DisplayName("buildGPTModel correctly assembles model from weight map with fused c_attn and weight tying")
    void testBuildGPTModel() {
        int vocabSize = 10;
        int contextLength = 8;
        int embDim = 4;
        int nHeads = 2;
        int nLayers = 2;
        int hiddenDim = 4 * embDim; // 16

        GPTConfig config = new GPTConfig(vocabSize, contextLength, embDim, nHeads, nLayers, true);
        Map<String, Tensor> weights = new HashMap<>();

        // Embeddings
        weights.put("wte", Tensor.zeros(vocabSize, embDim));
        weights.put("wpe", Tensor.zeros(contextLength, embDim));

        // 2 Blocks
        for (int l = 0; l < nLayers; l++) {
            String p = "blocks." + l + ".";
            weights.put(p + "ln_1.weight", Tensor.zeros(embDim));
            weights.put(p + "ln_1.bias", Tensor.zeros(embDim));

            // Fused c_attn: [4, 12] and [12]
            weights.put(p + "attn.c_attn.weight", Tensor.zeros(embDim, 3 * embDim));
            weights.put(p + "attn.c_attn.bias", Tensor.zeros(3 * embDim));

            weights.put(p + "attn.c_proj.weight", Tensor.zeros(embDim, embDim));
            weights.put(p + "attn.c_proj.bias", Tensor.zeros(embDim));

            weights.put(p + "ln_2.weight", Tensor.zeros(embDim));
            weights.put(p + "ln_2.bias", Tensor.zeros(embDim));

            weights.put(p + "mlp.c_fc.weight", Tensor.zeros(embDim, hiddenDim));
            weights.put(p + "mlp.c_fc.bias", Tensor.zeros(hiddenDim));

            weights.put(p + "mlp.c_proj.weight", Tensor.zeros(hiddenDim, embDim));
            weights.put(p + "mlp.c_proj.bias", Tensor.zeros(embDim));
        }

        // Final norm
        weights.put("ln_f.weight", Tensor.zeros(embDim));
        weights.put("ln_f.bias", Tensor.zeros(embDim));

        GPTModel model = WeightLoader.buildGPTModel(config, weights);
        assertNotNull(model);

        // Verify model can run forward pass
        int[][] input = {{1, 2, 3}};
        Tensor logits = model.forward(input);
        assertArrayEquals(new int[]{1, 3, vocabSize}, logits.shape());

        // Verify model can run generation
        int[][] generated = model.generate(input, 2, contextLength, 0.0f, null, null, null);
        assertEquals(5, generated[0].length);
    }
}
