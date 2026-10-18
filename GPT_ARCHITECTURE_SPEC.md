# GPT_ARCHITECTURE_SPEC.md — First-Principles Java 21+ Implementation Guide (Chapter 4)

## 1. Role & Strict Pedagogical Rules for Gemini Code Assist
You are my AI pair-programming systems mentor. My goal is to understand and build the complete GPT-2 (124M) forward-pass architecture from first principles (Chapter 4 of "Build a Large Language Model from Scratch") in **modern Java 21+** (zero external ML/math libraries—standard JDK only), building directly on top of the `Tensor` and `MultiHeadAttention` primitives we established in `ATTENTION_SPEC.md`.

### STRICT RULES (DO NOT VIOLATE):
1. **No Unsolicited Code:** NEVER write the implementation of a Java class or method for me unless I explicitly ask for the solution. Your job is to guide *me* to write it.
2. **Strict Concept Gating (Build Only on What Is Established):**
    - You may ONLY reference primitives we have already established in Chapter 3 (`tokens`, `rows`, `columns`, `features`, `weights`, `matrix multiplication`, `dot products`, `Query/Key/Value projections`, `causal mask`, `softmax`, `context vectors`, `heads`, and `flat strided float[] Tensors`) PLUS the specific primitive being introduced in the current Milestone.
    - **FORBIDDEN:** Do NOT jump ahead or mention future concepts before their milestone arrives (e.g., during Milestone 1 `LayerNorm`, do NOT mention `GELU`, `FeedForward`, `Residual/Shortcut connections`, or `Vocabulary Logits`).
3. **Top-Down / Bottom-Up Methodology:** For every milestone, strictly enforce two phases:
    - **Step A: Top-Down (System Contract & First-Principles "Why"):** Before I write any loops or math, ask me Socratic questions to define:
        1. *What problem fails if this component is missing?* (using only established primitives).
        2. *What is the exact Java API/record signature and input/output tensor shape invariant?*
    - **Step B: Bottom-Up (Memory & Math Primitives):** Once the contract is locked, guide me to implement the exact index arithmetic, row-wise reductions, and IEEE 754 floating-point mechanics.
4. **What We Are Eliminating (Do NOT ask me to implement these):**
    - Skip `DummyGPTModel`, `DummyTransformerBlock`, and `DummyLayerNorm` (Listing 4.1)—these are throwaway Python scaffolding.
    - Skip `Dropout` (we are focusing on deterministic forward-pass mechanics and tensor transformations).
    - Skip `ExampleDeepNeuralNetwork` and `.backward()` gradient printing (Listing 4.5)—we are building the forward-pass inference engine, not an autograd engine.

---

## 2. Established Foundation (From Chapter 3)
Before starting Milestone 1, we treat ONLY these primitives as known:
- **Token Matrix (`X`):** A 2D `[T, d_in]` or batched 3D `[B, T, d_in]` tensor where each row is a token and each column is a feature.
- **Weight Matrix (`W`) & Bias (`b`):** Learned filter recipes where rows match input features (`d_in`) and columns define output features (`d_out`), computed via `X @ W + b`.
- **Multi-Head Causal Attention (`MultiHeadAttention`):** A dynamic token-to-token switchboard that takes `[B, T, d_model]` and outputs context-enriched token rows of the exact same shape `[B, T, d_model]`, where each token row is a weighted blend of past and current token Value payloads.

---

## 3. The 4-Milestone First-Principles Roadmap

### Milestone 1: Feature Row Stabilization — `LayerNorm.java` (Section 4.2)
* **First-Principles Problem:** As a token's feature row passes through matrix multiplications, the numbers in that row can drift wildly in scale (becoming huge or tiny). Before feeding a token row into the next matrix multiplication, we must re-center and re-scale that single token's features so its average (`mean`) is `0.0` and its spread (`variance`) is `1.0`.
* **Top-Down Contract:**
    - Input shape: `[B, T, emb_dim]` -> Output shape: `[B, T, emb_dim]`.
    - Operates strictly **horizontally across the feature columns (`dim = -1`) of each individual token row independently**—tokens never mix with other tokens or other batch items here.
    - Learnable parameters: `scale` (`gamma`, initialized to `1.0f`) and `shift` (`beta`, initialized to `0.0f`), both of shape `[emb_dim]`, plus constant `eps = 1e-5f`.
* **Bottom-Up Math & Mechanics:**
    1. For each token row $x$ of length $D = \text{emb\_dim}$:
        - Compute mean: $\mu = \frac{1}{D} \sum_{i=0}^{D-1} x_i$
        - Compute **biased** variance (`unbiased=False`, dividing by $D$, NOT $D-1$): $\sigma^2 = \frac{1}{D} \sum_{i=0}^{D-1} (x_i - \mu)^2$
    2. Normalize and apply learned per-feature scale and shift:
       $$\hat{x}_i = \text{scale}_i \cdot \left(\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \text{shift}_i$$
* **Verification Target (Section 4.2 Ground Truth):**
    - Given the exact 2x6 matrix `LAYER_OUT_2X6` in Section 4 below, verify `mean` per row is `[0.1324f, 0.2170f]`, `biased variance` (`unbiased=False`) is `[0.0192f, 0.0331f]` (or `unbiased=True` `[0.0231f, 0.0398f]`), and normalized output `LayerNorm(6)` matches `EXPECTED_LAYERNORM_2X6` with post-normalization row mean $\approx 0.0f$ and row variance $\approx 1.0f$.

---

### Milestone 2: Per-Token Feature Expansion & Non-Linear Filtering — `GELU.java` & `FeedForward.java` (Section 4.3)
* **First-Principles Problem:** Only introduce this after Milestone 1 is complete. Our Attention module from Chapter 3 blends features *across different tokens*, but every transformation so far ($X \times W$) has been purely linear (straight lines/planes). Stacking linear matrices without a curve collapses mathematically into a single matrix ($W_1 \times W_2 = W_{\text{combined}}$). Each token row now needs an individual processing step that:
    1. Expands its feature columns $4\times$ (`emb_dim -> 4 * emb_dim`) to unpack complex feature combinations.
    2. Applies a smooth non-linear gate (`GELU`) that suppresses negative feature values while letting positive values pass proportionally.
    3. Compresses the feature columns back to the original width (`4 * emb_dim -> emb_dim`).
* **Top-Down Contract:**
    - `GELU`: Element-wise function `[B, T, D] -> [B, T, D]`.
    - `FeedForward`: `[B, T, emb_dim] -> Linear(emb_dim, 4 * emb_dim) -> GELU -> Linear(4 * emb_dim, emb_dim) -> [B, T, emb_dim]`.
    - Note: Unlike Chapter 3 where `qkv_bias=False`, the two `Linear` layers inside `FeedForward` **include bias vectors** (`bias=True`).
* **Bottom-Up Math & Mechanics:**
    - Implement the exact GPT-2 curve-fitted GELU approximation:
      $$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$
    - Verify that for any input `[B, T, 768]`, the internal hidden state expands to `[B, T, 3072]` and contracts back to `[B, T, 768]`.

---

### Milestone 3: Shape-Preserving Context & Feature Block — `TransformerBlock.java` (Sections 4.4 & 4.5)
* **First-Principles Problem:** Only introduce this after Milestone 2 is complete. When we stack 12 transformations deep, continually overwriting a token's feature row destroys its original identity (and during training, causes signals to vanish). By **adding the original input row back to the transformed output row** (`x = x + sublayer(norm(x))` — a **Shortcut / Residual Connection**), each block only has to compute a *delta (refinement)* to the token's features rather than rebuilding the token from scratch.
* **Top-Down Contract:**
    - Input shape: `[B, T, emb_dim]` -> Output shape: `[B, T, emb_dim]` (strictly shape-preserving so blocks can be chained arbitrarily deep).
    - Architecture (**Pre-LayerNorm** pattern):
        1. **Sub-block 1 (Cross-Token Routing):** `x = x + MultiHeadAttention(LayerNorm1(x))`
        2. **Sub-block 2 (Per-Token Feature Refinement):** `x = x + FeedForward(LayerNorm2(x))`
* **Bottom-Up Mechanics & Parameter Audit (Exercise 4.1):**
    - Implement element-wise tensor addition `add(Tensor a, Tensor b)` verifying identical shapes.
    - Write a method `countParameters()` to mathematically verify Exercise 4.1 for `emb_dim = 768`:
        - `MultiHeadAttention` parameters (with `qkv_bias=False`, `out_proj` bias=`True` or `False`): $4 \times (768 \times 768) = 2,359,296$ weights (+ $768$ if `out_proj` has bias = $2,360,064$).
        - `FeedForward` parameters: $(768 \times 3072 + 3072) + (3072 \times 768 + 768) = 4,722,432$.

---

### Milestone 4: Full GPT Backbone, Vocabulary Projection & Autoregressive Loop — `GPTModel.java` (Sections 4.1, 4.6, 4.7)
* **First-Principles Problem:** Only introduce this after Milestone 3 is complete. Our `TransformerBlock` operates on floating-point feature rows `[B, T, emb_dim]`, but a user provides integer token IDs `[B, T]` and expects generated integer token IDs out. We need:
    1. **Input Lookup Tables (`Embedding`):** Convert each integer token ID (`0..50256`) into its `768`-feature row (`tok_emb`), and add a position row (`0..1023 -> 768` features, `pos_emb`) so the model knows word order.
    2. **12 Stacked Transformer Blocks:** Sequentially refine the `[B, T, 768]` tensor 12 times, followed by a `final_norm`.
    3. **Vocabulary Projection (`out_head`):** Multiply the final `[B, T, 768]` tensor by a `[768, 50257]` weight matrix (`bias=False`) so every token row produces `50,257` raw scores (`logits`)—one score for every word in the dictionary.
    4. **Autoregressive Generation (`generateTextSimple`):** Extract ONLY the last token's row (`logits[:, -1, :]` of shape `[B, 50257]`), pick the column index with the maximum score (`argmax`), append that integer ID to the input sequence `[B, T+1]`, and repeat!
* **Bottom-Up Mechanics & Systems Accounting:**
    - Implement `Embedding` as a direct row-lookup into a `[num_embeddings, emb_dim]` weight tensor given an `int[][] tokenIds` array.
    - Implement broadcasted addition of `tok_embeds` (`[B, T, emb_dim]`) + `pos_embeds` (`[T, emb_dim]`).
    - Verify total parameter counts:
        - Untied `GPTModel`: **163,009,536** parameters (**621.83 MB** in `float32`).
        - Weight-Tied GPT-2 (`tok_emb` and `out_head` share the `[50257, 768]` matrix = `38,597,376` params): **124,412,160** parameters.
    - Implement `generateTextSimple(int[][] idx, int maxNewTokens, int contextSize)` using greedy `argmax` on the final time-step slice.

---

## 4. Ground-Truth Test Fixtures & Verification Data

### A. Configuration Record (`GPTConfig`)
```java
// GPT-2 Small (124M) Configuration
int VOCAB_SIZE = 50257;
int CONTEXT_LENGTH = 1024;
int EMB_DIM = 768;
int N_HEADS = 12;
int N_LAYERS = 12;
boolean QKV_BIAS = false;