package com.galt.llm;

import com.galt.llm.config.GPTConfig;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Live demonstration of the pure Java 21+ GPT-2 Architecture,
 * Evaluation Engine (Cross-Entropy & Perplexity), and Stochastic Decoding Engine.
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("========================================================================");
        System.out.println("   JAVA-LLM: First-Principles GPT-2 Engine (Pure Standard JDK 21+)     ");
        System.out.println("========================================================================");

        // 1. Architecture Specs
        GPTConfig gpt2Config = GPTConfig.gpt2_124M();
        System.out.println("\n[1] GPT-2 124M Hyperparameters:");
        System.out.println("    - Vocabulary Size : " + gpt2Config.vocabSize() + " tokens");
        System.out.println("    - Context Window  : " + gpt2Config.contextLength() + " tokens");
        System.out.println("    - Embedding Dim   : " + gpt2Config.embDim() + " features");
        System.out.println("    - Attention Heads : " + gpt2Config.nHeads() + " heads (head_dim = " + (gpt2Config.embDim() / gpt2Config.nHeads()) + ")");
        System.out.println("    - Number of Layers: " + gpt2Config.nLayers() + " Transformer blocks");
        System.out.println("    - FeedForward Dim : " + (4 * gpt2Config.embDim()) + " (4x expansion)");

        // 2. Demo Vocabulary Mapping (Section 5.3)
        // Mapping token IDs to human-readable words
        Map<Integer, String> vocab = new HashMap<>();
        vocab.put(0, "closer");
        vocab.put(1, "every");
        vocab.put(2, "effort");
        vocab.put(3, "forward");
        vocab.put(4, "inches");
        vocab.put(5, "moves");
        vocab.put(6, "pizza");
        vocab.put(7, "toward");
        vocab.put(8, "you");
        vocab.put(9, "<eos>");

        int vocabSize = 10;
        int contextLength = 16;
        int embDim = 16;
        int nHeads = 4;
        int nLayers = 2;

        GPTConfig demoConfig = new GPTConfig(vocabSize, contextLength, embDim, nHeads, nLayers, true);
        System.out.println("\n[2] Building Demo Transformer Model (Vocab = 10, Layers = 2)...");
        GPTModel model = buildDemoModel(demoConfig);
        System.out.println("    Model initialized successfully.");

        // 3. Prompt setup: "every effort moves you" -> [1, 2, 5, 8]
        int[][] prompt = {{1, 2, 5, 8}};
        System.out.println("\n[3] Input Prompt:");
        printTokens("    Prompt Tokens: ", prompt[0], vocab);

        // 4. Mode 1: Deterministic Greedy Decoding (temperature = 0.0)
        System.out.println("\n[4] Generation with Greedy Decoding (Temperature = 0.0, Argmax):");
        int[][] greedyOutput = model.generate(prompt, 5, contextLength, 0.0f, null, 9, null);
        printTokens("    Generated Output: ", greedyOutput[0], vocab);

        // 5. Mode 2: Stochastic Sampling with Temperature & Top-K
        float temperature = 0.8f;
        int topK = 3;
        Random rng = new Random(42);
        System.out.printf("\n[5] Generation with Stochastic Sampling (Temperature = %.1f, Top-K = %d):%n", temperature, topK);
        int[][] sampledOutput = model.generate(prompt, 5, contextLength, temperature, topK, 9, rng);
        printTokens("    Generated Output: ", sampledOutput[0], vocab);

        // 6. Evaluation Metric Demonstration
        System.out.println("\n[6] Evaluation Engine Demonstration:");
        Tensor logits = model.forward(prompt);
        // Ground truth targets: shifted inputs [2, 5, 8, 3] ("effort", "moves", "you", "forward")
        int[][] targets = {{2, 5, 8, 3}};
        float loss = LossUtils.crossEntropyLoss(logits, targets);
        float perplexity = LossUtils.perplexity(loss);
        System.out.printf("    - Evaluated Tokens   : %d tokens%n", targets.length * targets[0].length);
        System.out.printf("    - Cross-Entropy Loss : %.4f%n", loss);
        System.out.printf("    - Perplexity (exp)   : %.2f (words model is torn between)%n", perplexity);

        System.out.println("\n========================================================================");
        System.out.println("   All 3 Chapters & Specs (Attention, GPT-2, Decoding) Complete!       ");
        System.out.println("========================================================================");
    }

    private static void printTokens(String label, int[] tokens, Map<Integer, String> vocab) {
        StringBuilder sb = new StringBuilder();
        sb.append(label).append("[ ");
        for (int i = 0; i < tokens.length; i++) {
            int t = tokens[i];
            String word = vocab.getOrDefault(t, "token_" + t);
            sb.append("\"").append(word).append("\"");
            if (i < tokens.length - 1) {
                sb.append(", ");
            }
        }
        sb.append(" ]");
        System.out.println(sb);
    }

    private static GPTModel buildDemoModel(GPTConfig config) {
        int vocabSize = config.vocabSize();
        int contextLength = config.contextLength();
        int embDim = config.embDim();
        int nHeads = config.nHeads();
        int nLayers = config.nLayers();
        int hiddenDim = 4 * embDim;

        Map<String, Tensor> weights = new HashMap<>();

        // Embeddings
        float[] tokData = new float[vocabSize * embDim];
        for (int i = 0; i < tokData.length; i++) {
            tokData[i] = (float) Math.sin(i * 0.1f) * 0.1f;
        }
        weights.put("wte", Tensor.of(tokData, vocabSize, embDim));

        float[] posData = new float[contextLength * embDim];
        for (int i = 0; i < posData.length; i++) {
            posData[i] = (float) Math.cos(i * 0.1f) * 0.05f;
        }
        weights.put("wpe", Tensor.of(posData, contextLength, embDim));

        // Transformer blocks
        for (int l = 0; l < nLayers; l++) {
            String p = "blocks." + l + ".";

            float[] ones = new float[embDim];
            java.util.Arrays.fill(ones, 1.0f);
            float[] zeros = new float[embDim];

            weights.put(p + "ln_1.weight", Tensor.of(ones.clone(), embDim));
            weights.put(p + "ln_1.bias", Tensor.of(zeros.clone(), embDim));

            // Fused c_attn [embDim, 3 * embDim]
            float[] attnW = new float[embDim * 3 * embDim];
            for (int i = 0; i < attnW.length; i++) {
                attnW[i] = ((i % 13) - 6) * 0.02f;
            }
            weights.put(p + "attn.c_attn.weight", Tensor.of(attnW, embDim, 3 * embDim));

            float[] attnB = new float[3 * embDim];
            weights.put(p + "attn.c_attn.bias", Tensor.of(attnB, 3 * embDim));

            // Projection [embDim, embDim]
            float[] projW = new float[embDim * embDim];
            for (int i = 0; i < projW.length; i++) {
                projW[i] = ((i % 11) - 5) * 0.02f;
            }
            weights.put(p + "attn.c_proj.weight", Tensor.of(projW, embDim, embDim));
            weights.put(p + "attn.c_proj.bias", Tensor.of(new float[embDim], embDim));

            weights.put(p + "ln_2.weight", Tensor.of(ones.clone(), embDim));
            weights.put(p + "ln_2.bias", Tensor.of(zeros.clone(), embDim));

            // FeedForward
            float[] fcW = new float[embDim * hiddenDim];
            for (int i = 0; i < fcW.length; i++) {
                fcW[i] = ((i % 7) - 3) * 0.02f;
            }
            weights.put(p + "mlp.c_fc.weight", Tensor.of(fcW, embDim, hiddenDim));
            weights.put(p + "mlp.c_fc.bias", Tensor.of(new float[hiddenDim], hiddenDim));

            float[] projFcW = new float[hiddenDim * embDim];
            for (int i = 0; i < projFcW.length; i++) {
                projFcW[i] = ((i % 5) - 2) * 0.02f;
            }
            weights.put(p + "mlp.c_proj.weight", Tensor.of(projFcW, hiddenDim, embDim));
            weights.put(p + "mlp.c_proj.bias", Tensor.of(new float[embDim], embDim));
        }

        // Final norm
        float[] ones = new float[embDim];
        java.util.Arrays.fill(ones, 1.0f);
        weights.put("ln_f.weight", Tensor.of(ones, embDim));
        weights.put("ln_f.bias", Tensor.of(new float[embDim], embDim));

        return WeightLoader.buildGPTModel(config, weights);
    }
}
