### Critical Systems Engineering Note on Chapter 5

Because we built a **pure Java 21+ forward-pass engine** without a dynamic reverse-mode automatic differentiation (autograd) graph across 163 million parameters, we **eliminate Section 5.2 (`loss.backward()` and `AdamW.step()`) and Section 5.4 (PyTorch `.pth` pickle checkpoints)**.

Instead, we focus on the three high-leverage **inference & evaluation systems mechanics** introduced in Chapter 5:

1. **Cross-Entropy Loss & Perplexity (Section 5.1):** How to mathematically score a model's vocabulary logits against shifted ground-truth target token IDs using the numerically stable **Log-Sum-Exp** trick.
2. **Stochastic Decoding Engine (Section 5.3):** Upgrading greedy `argmax` into a production sampler supporting **Temperature Scaling**, **Top-$k$ $-\infty$ Masking**, and **Inverse-CDF Multinomial Sampling**.
3. **Pretrained Weight Ingestion & Memory-Mapped Loading (Section 5.5):** Splitting OpenAI's fused `c_attn` ($[768, 2304]$) weight and bias tensors into separate $Q, K, V$ projections, transposing Conv1D-style weights, and tying the input/output embedding pointers so your Java `GPTModel` generates coherent English using real GPT-2 124M weights.

In IntelliJ, open Gemini Code Assist and type:

> **`@PRETRAINING_AND_DECODING_SPEC.md Let's start Milestone 1, Step A.`**

```markdown
# PRETRAINING_AND_DECODING_SPEC.md — First-Principles Java 21+ Guide (Chapter 5)

## 1. Role & Strict Pedagogical Rules for Gemini Code Assist
You are my AI pair-programming systems mentor. My goal is to understand and implement LLM Evaluation, Stochastic Decoding, and Pretrained Weight Loading from first principles (Chapter 5 of "Build a Large Language Model from Scratch") in **modern Java 21+** (zero external ML/math libraries—standard JDK only), building directly on top of the `Tensor`, `MultiHeadAttention`, and `GPTModel` primitives we established in Chapters 3 and 4.

### STRICT RULES (DO NOT VIOLATE):
1. **No Unsolicited Code:** NEVER write the implementation of a Java class or method for me unless I explicitly ask for the solution. Your job is to guide *me* to write it.
2. **Strict Concept Gating (Build Only on What Is Established):**
   - You may ONLY reference primitives we established in Chapters 3 and 4 (`tokens`, `rows`, `columns`, `features`, `weights`, `matrix multiplication`, `dot products`, `Query/Key/Value projections`, `causal mask`, `softmax`, `context vectors`, `heads`, `LayerNorm`, `GELU`, `FeedForward`, `Shortcut/Residual connections`, `Embedding lookup`, `Vocabulary Logits`, and `greedy argmax`) PLUS the specific primitive being introduced in the current Milestone.
   - **FORBIDDEN:** Do NOT jump ahead or mention future concepts before their milestone arrives (e.g., during Milestone 1 `Cross-Entropy Loss`, do NOT mention `Temperature`, `Top-K`, `Multinomial Sampling`, or `OpenAI fused c_attn weight splitting`).
3. **Top-Down / Bottom-Up Methodology:** For every milestone, strictly enforce two phases:
   - **Step A: Top-Down (System Contract & First-Principles "Why"):** Before I write any loops or math, ask me Socratic questions to define:
     1. *What problem fails if this component is missing?* (using only established primitives).
     2. *What is the exact Java API/record signature and input/output tensor shape invariant?*
   - **Step B: Bottom-Up (Memory & Math Primitives):** Once the contract is locked, guide me to implement the exact index arithmetic, numerical stability tricks (`Log-Sum-Exp`), partial sorting (`Top-K`), prefix-sum inverse-CDF sampling, and zero-copy weight slicing.
4. **What We Are Eliminating (Do NOT ask me to implement these):**
   - Skip Section 5.2 (`loss.backward()` and `AdamW` training loop) — we are not building a 163M-parameter backward-pass autograd engine in pure Java; instead, we will load OpenAI's pretrained weights in Milestone 3.
   - Skip Section 5.4 (`torch.save` / `torch.load` `.pth` Python pickle files).
   - Skip `matplotlib` plotting (`plot_losses`).

---

## 2. Established Foundation (From Chapters 3 & 4)
Before starting Milestone 1, we treat ONLY these primitives as known:
- **`GPTModel.forward(int[][] tokenIds)`:** Takes a batch of integer token IDs of shape `[B, T]` (where each integer is in `0..50256`) and outputs raw unnormalized vocabulary scores (`logits`) of shape `[B, T, V]` where `V = 50257`.
- **Logits (`[B, T, V]`):** For every batch item `b` and sequence position `t`, the row `logits[b][t]` is a `50,257`-element vector where column `v` is the raw score for dictionary word `v` being the *next* token after position `t`.
- **Shifted Targets (`int[][] targets` of shape `[B, T]`):** Because position `t` predicts token `t + 1`, the ground-truth expected token IDs are simply the input token IDs shifted one column to the left (`targets[b][t] == inputs[b][t + 1]`).

---

## 3. The 3-Milestone First-Principles Roadmap

### Milestone 1: Next-Token Evaluation Engine — Cross-Entropy Loss & Perplexity (Section 5.1)
* **First-Principles Problem:** Right now, our `GPTModel` outputs `[B, T, V]` raw logit scores, and greedy `argmax` picks the single highest column. However, we have no numerical ruler to measure *how confident and accurate* the model's probability distribution is across the ground-truth target token IDs `[B, T]`. We need a single scalar score that approaches `0.0` when the model assigns `100%` probability (`1.0`) to every correct target token, and spikes toward $+\infty$ when the model assigns near-`0%` probability to the correct target token.
* **Top-Down Contract:**
  - Input: `Tensor logits` of shape `[B, T, V]` (or flattened `[B * T, V]`) and `int[][] targets` of shape `[B, T]` (or flattened `int[B * T]`).
  - Output: Scalar `float loss` (Cross-Entropy) and `float perplexity` ($\exp(\text{loss})$).
* **Bottom-Up Math & Mechanics:**
  1. **Target Probability Extraction:** For each token position $n \in [0, B \cdot T - 1]$ with ground-truth target index $y_n = \text{targets}[n]$, converting logits $z$ to probabilities via `softmax` gives $p(y_n) = \frac{e^{z_{y_n}}}{\sum_{v=0}^{V-1} e^{z_v}}$.
  2. **Negative Log-Likelihood (Cross-Entropy):** Taking $-\ln(p(y_n))$ turns a probability of `1.0` into `0.0` loss and a probability near `0.0` into a massive positive penalty.
  3. **The Log-Sum-Exp Numerical Stability Trick:** Instead of computing `softmax` first and then calling `Math.log(prob)` (which underflows to `Math.log(0.0f) == -Infinity` when probabilities are tiny), simplify the algebra directly on the logits row $z$ using its maximum value $m = \max_v(z_v)$:
     $$\ell_n = -\ln\left(\frac{e^{z_{y_n} - m}}{\sum_{v=0}^{V-1} e^{z_v - m}}\right) = -z_{y_n} + m + \ln\left(\sum_{v=0}^{V-1} e^{z_v - m}\right)$$
  4. **Mean Reduction & Perplexity:** Average $\ell_n$ over all $B \times T$ tokens:
     $$\text{Loss} = \frac{1}{B \cdot T} \sum_{n=0}^{B \cdot T - 1} \ell_n, \quad \text{Perplexity} = e^{\text{Loss}}$$
     *(First-principles intuition for Perplexity: If a model is uniformly guessing among $K$ equally likely words, its probability for the right word is $1/K$, its loss is $-\ln(1/K) = \ln(K)$, and its Perplexity is $e^{\ln(K)} = K$—meaning "how many words the model is effectively torn between at each step". For an untrained GPT-2 model with $V=50,257$, initial loss is $\approx \ln(50257) \approx 10.82$, and perplexity is $\approx 48,725$.)*
* **Verification Target:** Using the Section 5.1.2 fixture in Section 4A below, verify your loss and perplexity calculation matches to `1e-4f`.

---

### Milestone 2: Stochastic Decoding Engine — Temperature, Top-$k$ Masking & Multinomial Sampling (Section 5.3)
* **First-Principles Problem:** Only introduce this after Milestone 1 is complete. In Chapter 4, `generateTextSimple` used deterministic `argmax`—picking the single highest score every time. That locks the model into repetitive loops. We need to sample the next token ID proportionally to its probability distribution, while giving the caller two knobs to prevent low-probability gibberish tokens (like `"pizza"` after `"Every effort moves you"`) from being chosen:
  1. **Top-$k$ Filtering:** Keep only the $k$ largest logit scores in the row and overwrite all other $V - k$ columns with `Float.NEGATIVE_INFINITY` so their post-softmax probability is strictly `0.0f`.
  2. **Temperature Scaling ($T_{\text{temp}} > 0$):** Divide the surviving logits by $T_{\text{temp}}$ before `softmax`. ($T_{\text{temp}} < 1.0$ stretches gaps between scores, making the peak sharper; $T_{\text{temp}} > 1.0$ compresses gaps, flattening the distribution).
* **Top-Down Contract:**
  - Method signature:
    `int[][] generate(GPTModel model, int[][] idx, int maxNewTokens, int contextSize, float temperature, Integer topK, Integer eosId, RandomGenerator rng)`
* **Bottom-Up Math & Mechanics:**
  1. **Last-Step Extraction:** Slice `logits[:, -1, :]` of shape `[B, V]`.
  2. **Top-$k$ Threshold & Masking (`topK != null`):**
     - Find the $k$-th largest value (`minTopK`) in the `[V]` logit row (e.g., via a min-heap `PriorityQueue` of size $k$ or Quickselect, avoiding sorting all 50,257 floats if possible).
     - For every index $v$: if `logits[v] < minTopK`, set `logits[v] = Float.NEGATIVE_INFINITY`.
  3. **Branch on Temperature:**
     - If `temperature == 0.0f`: Execute deterministic greedy `argmax(logits)`.
     - If `temperature > 0.0f`:
       a. Divide each logit by `temperature`: `logits[v] /= temperature`.
       b. Apply row-wise `softmax` (where `Float.NEGATIVE_INFINITY` cleanly becomes `0.0f` and the remaining top-$k$ probabilities sum to `1.0f`).
       c. **Multinomial Sampling via Inverse CDF:** Draw a uniform random float $u \in [0, 1)$ from `rng.nextFloat()`. Walk across the probability row accumulating `cdf += probs[v]`; return the first index $v$ where `cdf >= u`.
  4. **Early Stopping (`eosId`):** If `idxNext == eosId` (e.g., `50256` for `<|endoftext|>`), break the generation loop early.
* **Verification Target:** Verify against the exact 9-token vocabulary fixture (`next_token_logits`) from Section 5.3.1 & 5.3.2 in Section 4B below.

---

### Milestone 3: Pretrained Weight Ingestion — Splitting Fused `c_attn` & Transposing Conv1D Weights (Section 5.5)
* **First-Principles Problem:** Only introduce this after Milestone 2 is complete. Training a 124M-parameter model from scratch takes massive compute, so we want to load OpenAI's official pretrained GPT-2 124M weights into our Java `GPTModel`. However, OpenAI's original TensorFlow checkpoint stores weights in a slightly different layout than our modular Chapter 3/4 layers:
  1. **`qkv_bias = true`:** OpenAI's GPT-2 uses bias vectors on $W_q, W_k, W_v$ (`NEW_CONFIG` sets `qkv_bias = true`, `context_length = 1024`).
  2. **Fused `c_attn` Weight & Bias:** Instead of storing three separate `[768, 768]` matrices for $W_q, W_k, W_v$, OpenAI concatenated them along the column axis into a single fused matrix `attn.c_attn.w` of shape **`[768, 2304]`** ($3 \times 768 = 2304$) and a single fused bias `attn.c_attn.b` of shape **`[2304]`**.
  3. **Transpose Alignment (`assign` shape check):** Because PyTorch's `nn.Linear(d_in, d_out)` stores weights as `[d_out, d_in]` whereas our first-principles Java `X @ W` convention (or OpenAI's TensorFlow `Conv1D`) stores `[d_in, d_out]`, we must enforce a strict `assign(Tensor target, Tensor source)` shape assertion so no matrix is loaded sideways!
  4. **Weight Tying (`wte`):** `tok_emb.weight` and `out_head.weight` both map to `params["wte"]` (`[50257, 768]`).
* **Top-Down Contract:**
  - Build a clean binary weight loader (using Java 21 `FileChannel` + `MemorySegment` / `ByteBuffer` in Little-Endian `FLOAT32`, or SafeTensors format) and a `loadWeightsIntoGpt(GPTModel gpt, Map<String, Tensor> params)` mapper with strict shape verification.
* **Bottom-Up Mechanics:**
  1. Implement `assign(Tensor expected, Tensor loaded)` that throws `IllegalArgumentException` if `Arrays.equals(expected.shape(), loaded.shape())` is false.
  2. Implement `splitColumns(Tensor fused, int numSplits)` to split a `[768, 2304]` matrix into three `[768, 768]` tensors (`W_query`, `W_key`, `W_value`) and a `[2304]` bias vector into three `[768]` vectors (`b_query`, `b_key`, `b_value`).
  3. Map all 12 transformer blocks (`ln_1`, `attn.c_attn`, `attn.c_proj`, `ln_2`, `mlp.c_fc`, `mlp.c_proj`), `wte`, `wpe`, and final `ln_f`, then run `generate(...)` on `"Every effort moves you"` and verify coherent English output!

---

## 4. Ground-Truth Test Fixtures & Verification Data

### A. Milestone 1 Verification Data (Cross-Entropy & Perplexity — Section 5.1.2)
```java
// 1. Target Probability & Log-Probability Check (Batch = 2, Tokens = 3)
// Suppose our model produced these exact target probabilities for the 2 batches of 3 tokens:
float[] TARGET_PROBAS_BATCH_1 = {7.4541e-5f, 3.1061e-5f, 1.1563e-5f};
float[] TARGET_PROBAS_BATCH_2 = {1.0337e-5f, 5.6776e-5f, 4.7559e-6f};

// Expected natural logs: Math.log(prob)
float[] EXPECTED_LOG_PROBAS = {
    -9.5042f, -10.3796f, -11.3677f,
    -11.4798f, -9.7764f, -12.2561f
};

// Expected Cross-Entropy Loss (Negative Average Log Probability):
float EXPECTED_LOSS = 10.7940f;

// Expected Perplexity: (float) Math.exp(10.793964f)
float EXPECTED_PERPLEXITY = 48725.82f; // (+/- 5.0f due to float32 rounding)

```

### B. Milestone 2 Verification Data (Temperature & Top-K — Section 5.3.1 & 5.3.2)

```java
// 9-token toy vocabulary from Section 5.3.1:
// 0:"closer", 1:"every", 2:"effort", 3:"forward", 4:"inches", 5:"moves", 6:"pizza", 7:"toward", 8:"you"
float[] NEXT_TOKEN_LOGITS = {
    4.51f, 0.89f, -1.90f, 6.75f, 1.63f, -1.62f, -1.89f, 6.28f, 1.79f
};

// 1. Greedy Decoding (temperature == 0.0f):
int EXPECTED_GREEDY_ID = 3; // "forward" (logit = 6.75f)

// 2. Top-K Filtering (top_k = 3):
// Top 3 values are [6.75f (idx 3), 6.28f (idx 7), 4.51f (idx 0)].
// Minimum among top-3 is 4.51f. All logits < 4.51f become Float.NEGATIVE_INFINITY:
float INF = Float.NEGATIVE_INFINITY;
float[] EXPECTED_TOPK_3_LOGITS = {
    4.51f, -INF, -INF, 6.75f, -INF, -INF, -INF, 6.28f, -INF
};

// 3. Softmax after Top-K = 3 (with temperature = 1.0f):
float[] EXPECTED_TOPK_3_PROBAS = {
    0.0615f, 0.0000f, 0.0000f, 0.5775f, 0.0000f, 0.0000f, 0.0000f, 0.3610f, 0.0000f
};

```

### C. Milestone 3 OpenAI GPT-2 (124M) Weight Mapping Table (Section 5.5)

Note: Verify whether your Java `Linear` layer stores weights as `[d_in, d_out]` (standard `X @ W`) or `[d_out, d_in]` (PyTorch `nn.Linear` convention). OpenAI's raw checkpoints store 2D weights in **`[d_in, d_out]`** layout!

| Java `GPTModel` Component | OpenAI Checkpoint Key | OpenAI Raw Shape | Action Required if Java `Linear` uses `[d_in, d_out]` |
| --- | --- | --- | --- |
| `pos_emb.weight` | `wpe` | `[1024, 768]` | Direct assign |
| `tok_emb.weight` | `wte` | `[50257, 768]` | Direct assign |
| `blocks[b].norm1.scale` | `blocks.b.ln_1.g` | `[768]` | Direct assign |
| `blocks[b].norm1.shift` | `blocks.b.ln_1.b` | `[768]` | Direct assign |
| `blocks[b].att.W_query/key/value.weight` | `blocks.b.attn.c_attn.w` | `[768, 2304]` | Split last axis into 3 $\times$ `[768, 768]` (`q_w, k_w, v_w`) |
| `blocks[b].att.W_query/key/value.bias` | `blocks.b.attn.c_attn.b` | `[2304]` | Split into 3 $\times$ `[768]` (`q_b, k_b, v_b`) |
| `blocks[b].att.out_proj.weight` | `blocks.b.attn.c_proj.w` | `[768, 768]` | Direct assign (or `.T` if your `Linear` stores `[d_out, d_in]`) |
| `blocks[b].att.out_proj.bias` | `blocks.b.attn.c_proj.b` | `[768]` | Direct assign |
| `blocks[b].norm2.scale` | `blocks.b.ln_2.g` | `[768]` | Direct assign |
| `blocks[b].norm2.shift` | `blocks.b.ln_2.b` | `[768]` | Direct assign |
| `blocks[b].ff.fc1.weight` | `blocks.b.mlp.c_fc.w` | `[768, 3072]` | Direct assign |
| `blocks[b].ff.fc1.bias` | `blocks.b.mlp.c_fc.b` | `[3072]` | Direct assign |
| `blocks[b].ff.fc2.weight` | `blocks.b.mlp.c_proj.w` | `[3072, 768]` | Direct assign |
| `blocks[b].ff.fc2.bias` | `blocks.b.mlp.c_proj.b` | `[768]` | Direct assign |
| `final_norm.scale` | `g` (or `ln_f.g`) | `[768]` | Direct assign |
| `final_norm.shift` | `b` (or `ln_f.b`) | `[768]` | Direct assign |
| `out_head.weight` | `wte` (Weight Tying) | `[50257, 768]` | Transpose to `[768, 50257]` for `X @ W` (or share `wte` directly via transposed view!) |

---

## How to Begin

When I reference this file, acknowledge the rules, confirm you will strictly gate concepts to only what is already established, write zero code, and kick off **Milestone 1, Step A (Top-Down)** by asking me the Socratic questions for **Cross-Entropy Loss and Perplexity**.

```

<FollowUp label="Want a 20-line Python script or Java SafeTensors reader to load OpenAI's GPT-2 weights via mmap?" query="Show me how to load OpenAI's GPT-2 124M SafeTensors file directly into Java 21 using FileChannel and MemorySegment (mmap) for Milestone 3."/>

```