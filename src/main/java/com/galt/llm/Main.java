package com.galt.llm;

import com.galt.llm.config.GPTConfig;

import java.util.*;

/**
 * Live demonstration and Interactive CLI REPL for the pure Java 21+ GPT-2 Engine.
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

        // 2. Demo Vocabulary Mapping
        Map<Integer, String> idToWord = new HashMap<>();
        Map<String, Integer> wordToId = new HashMap<>();
        registerToken(idToWord, wordToId, 0, "closer");
        registerToken(idToWord, wordToId, 1, "every");
        registerToken(idToWord, wordToId, 2, "effort");
        registerToken(idToWord, wordToId, 3, "forward");
        registerToken(idToWord, wordToId, 4, "inches");
        registerToken(idToWord, wordToId, 5, "moves");
        registerToken(idToWord, wordToId, 6, "pizza");
        registerToken(idToWord, wordToId, 7, "toward");
        registerToken(idToWord, wordToId, 8, "you");
        registerToken(idToWord, wordToId, 9, "<eos>");

        int vocabSize = 10;
        int contextLength = 16;
        int embDim = 16;
        int nHeads = 4;
        int nLayers = 2;

        GPTConfig demoConfig = new GPTConfig(vocabSize, contextLength, embDim, nHeads, nLayers, true);
        System.out.println("\n[2] Building Demo Transformer Model (Vocab = 10, Layers = 2)...");
        GPTModel model = buildDemoModel(demoConfig);
        System.out.println("    Model initialized successfully.");

        // 3. Automated Benchmark & Verification Demo
        System.out.println("\n[3] Automated Benchmark Demo (Prompt: 'every effort moves you'):");
        int[][] prompt = {{1, 2, 5, 8}};
        printTokens("    Prompt Tokens: ", prompt[0], idToWord);

        int[][] greedyOutput = model.generate(prompt, 5, contextLength, 0.0f, null, 9, null);
        printTokens("    Greedy (temp=0.0): ", greedyOutput[0], idToWord);

        Random rng = new Random(42);
        int[][] sampledOutput = model.generate(prompt, 5, contextLength, 0.8f, 3, 9, rng);
        printTokens("    Sampled (temp=0.8, top_k=3): ", sampledOutput[0], idToWord);

        Tensor logits = model.forward(prompt);
        int[][] targets = {{2, 5, 8, 3}};
        float loss = LossUtils.crossEntropyLoss(logits, targets);
        float perplexity = LossUtils.perplexity(loss);
        System.out.printf("    Loss: %.4f | Perplexity: %.2f%n", loss, perplexity);

        // 4. Interactive Command-Line REPL
        // Checks if an interactive console or stdin stream is available
        Scanner scanner = new Scanner(System.in);
        if (System.console() != null || (args.length > 0 && args[0].equalsIgnoreCase("--interactive"))) {
            runInteractiveRepl(model, contextLength, idToWord, wordToId, scanner);
        } else {
            System.out.println("\n[Tip] To launch interactive live chat mode in terminal, run with: --interactive");
        }

        System.out.println("\n========================================================================");
        System.out.println("   All 3 Chapters & Specs (Attention, GPT-2, Decoding) Complete!       ");
        System.out.println("========================================================================");
    }

    private static void registerToken(Map<Integer, String> idToWord, Map<String, Integer> wordToId, int id, String word) {
        idToWord.put(id, word);
        wordToId.put(word.toLowerCase(), id);
    }

    private static void runInteractiveRepl(
            GPTModel model,
            int contextLength,
            Map<Integer, String> idToWord,
            Map<String, Integer> wordToId,
            Scanner scanner
    ) {
        System.out.println("\n========================================================================");
        System.out.println("               LIVE INTERACTIVE GENERATION REPL                         ");
        System.out.println("========================================================================");
        System.out.println("Available vocabulary words: " + wordToId.keySet());
        System.out.println("Commands:");
        System.out.println("  /temp <float>   : Set temperature (e.g. /temp 0.8 or /temp 0.0 for greedy)");
        System.out.println("  /topk <int>     : Set top-k (e.g. /topk 3, or /topk 0 to disable)");
        System.out.println("  /tokens <int>   : Set max new tokens to generate (e.g. /tokens 6)");
        System.out.println("  exit / quit     : Exit the session");
        System.out.println("------------------------------------------------------------------------");

        float temperature = 0.8f;
        Integer topK = 3;
        int maxNewTokens = 5;
        Random rng = new Random();

        while (true) {
            System.out.print("\n[User Prompt] > ");
            if (!scanner.hasNextLine()) {
                break;
            }
            String line = scanner.nextLine().trim();
            if (line.isEmpty()) {
                continue;
            }
            if (line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
                System.out.println("Exiting interactive session.");
                break;
            }

            // Command handling
            if (line.startsWith("/temp")) {
                try {
                    temperature = Float.parseFloat(line.substring(5).trim());
                    System.out.printf("Temperature updated to: %.2f%n", temperature);
                } catch (Exception e) {
                    System.out.println("Usage: /temp <float> (e.g. /temp 0.7)");
                }
                continue;
            }
            if (line.startsWith("/topk")) {
                try {
                    int k = Integer.parseInt(line.substring(5).trim());
                    topK = (k > 0) ? k : null;
                    System.out.println("Top-K updated to: " + (topK != null ? topK : "disabled"));
                } catch (Exception e) {
                    System.out.println("Usage: /topk <int> (e.g. /topk 3)");
                }
                continue;
            }
            if (line.startsWith("/tokens")) {
                try {
                    maxNewTokens = Integer.parseInt(line.substring(7).trim());
                    System.out.println("Max new tokens updated to: " + maxNewTokens);
                } catch (Exception e) {
                    System.out.println("Usage: /tokens <int> (e.g. /tokens 6)");
                }
                continue;
            }

            // Tokenize words
            String[] words = line.split("\\s+");
            List<Integer> tokenList = new ArrayList<>();
            for (String w : words) {
                String clean = w.replaceAll("[^a-zA-Z<>]", "").toLowerCase();
                Integer id = wordToId.get(clean);
                if (id != null) {
                    tokenList.add(id);
                } else {
                    System.out.println("Warning: Word '" + w + "' not in vocabulary. Skipping.");
                }
            }

            if (tokenList.isEmpty()) {
                System.out.println("Error: No recognized tokens found in prompt. Try words like 'every effort moves you'");
                continue;
            }

            int[][] promptIdx = new int[1][tokenList.size()];
            for (int i = 0; i < tokenList.size(); i++) {
                promptIdx[0][i] = tokenList.get(i);
            }

            System.out.printf("Generating %d tokens (temp=%.2f, topK=%s)...%n", maxNewTokens, temperature, topK);
            int[][] result = model.generate(promptIdx, maxNewTokens, contextLength, temperature, topK, 9, rng);

            printTokens("[GPT Output] -> ", result[0], idToWord);
        }
    }

    private static void printTokens(String label, int[] tokens, Map<Integer, String> vocab) {
        StringBuilder sb = new StringBuilder();
        sb.append(label);
        for (int i = 0; i < tokens.length; i++) {
            int t = tokens[i];
            String word = vocab.getOrDefault(t, "token_" + t);
            sb.append(word);
            if (i < tokens.length - 1) {
                sb.append(" ");
            }
        }
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
