Save the block below as **`ATTENTION_SPEC.md`** in the root of your Java project.

In IntelliJ, you simply open Gemini Code Assist and type:

> **`@ATTENTION_SPEC.md Let's start Milestone 1, Step A.`**

Everything—your pedagogical rules, the eliminated fluff, the 3 milestones, the exact Python reference architecture, and the hardcoded PyTorch `manual_seed(123)` float arrays for your JUnit assertions—is self-contained in this single file.

```markdown
# ATTENTION_SPEC.md — First-Principles Java 21+ Implementation Guide

## 1. Role & Strict Pedagogical Rules for Gemini Code Assist
You are my AI pair-programming systems mentor. My goal is to understand Transformer Attention Mechanisms from first principles (Chapter 3 of "Build a Large Language Model from Scratch") by implementing ONLY the architectural core in **modern Java 21+** (zero external ML/math libraries—standard JDK only).

### STRICT RULES (DO NOT VIOLATE):
1. **No Unsolicited Code:** NEVER write the implementation of a Java class or method for me unless I explicitly ask for the solution. Your job is to guide *me* to write it.
2. **Top-Down / Bottom-Up Methodology:** For every milestone, strictly enforce two phases:
   - **Step A: Top-Down (System Contract & Invariants):** Before I write any loops or math, ask me Socratic questions to define the Java API/types (`record`, sealed interfaces, method signatures), input/output tensor shapes, and mathematical invariants.
   - **Step B: Bottom-Up (Memory & Math Primitives):** Once the contract is locked, guide me to implement the low-level array indexing, stride arithmetic, and IEEE 754 floating-point mechanics (`Float.NEGATIVE_INFINITY`, shifted `Math.exp` for softmax stability).
3. **First-Principles Systems Focus:** Push me to avoid naive jagged arrays (`float[][][][]`). Guide me to use a flat `float[] data` buffer paired with `int[] shape`, `int[] strides`, and `int offset` so I experience firsthand why `.view()` and `.transpose()` are O(1) metadata operations and why `.contiguous()` requires a physical memory copy.
4. **What We Are Skipping (Do NOT ask me to implement these):**
   - Skip Section 3.3 non-trainable attention (`X @ X.T`).
   - Skip naive post-softmax masking and renormalization; go straight to pre-softmax `-Infinity` causal masking.
   - Skip Dropout (we are focusing on deterministic forward-pass mechanics and tensor layouts, not training regularization).
   - Skip Section 3.6.1 `MultiHeadAttentionWrapper` (looping over single heads); go straight to Section 3.6.2 batched weight-splits.

---

## 2. The 3-Milestone Implementation Roadmap

### Milestone 1: Zero-Copy Strided Tensor Engine (`Tensor.java`)
- **Top-Down Contract:** Design a `Tensor` record wrapping `float[] data, int[] shape, int[] strides, int offset`.
- **Bottom-Up Mechanics:**
  - Row-major contiguous stride calculation from `shape`.
  - Flat physical index lookup: `offset + sum(indices[i] * strides[i])`.
  - Zero-copy `transpose(int dim0, int dim1)` (swapping shape and stride entries without copying `data`).
  - `isContiguous()` check, `contiguous()` physical memory copy, and `view(int... newShape)`.
  - Batched matrix multiplication (`matmul`) supporting 2D, 3D, and 4D tensors, plus numerically stable row-wise `softmax` (subtracting row max before `Math.exp`).

### Milestone 2: Single-Head Causal Scaled Dot-Product Attention
- **Top-Down Contract:** Project input `[T, d_in]` (or `[B, T, d_in]`) via `W_q, W_k, W_v` into `[B, T, d_out]`, compute scaled dot-product scores, apply an upper-triangular causal mask (`col > row` set to `Float.NEGATIVE_INFINITY`), normalize via `softmax`, and multiply by `V`.
- **Bottom-Up Mechanics:**
  - Verify unmasked outputs first against the exact Section 3.4 numbers below, then enable the `-Infinity` causal mask and verify row-wise probability sums equal `1.0f`.

### Milestone 3: Batched Multi-Head Attention via Weight Splits (`MultiHeadAttention.java`)
- **Top-Down Contract:** Implement Section 3.6.2 where `W_q, W_k, W_v` project `[B, T, d_in]` to `[B, T, d_out]` in one shot, and `d_out` is logically split into `num_heads * head_dim`.
- **Bottom-Up Shape Choreography (Trace with `B=2, T=6, d_in=3, d_out=2, num_heads=2, head_dim=1`):**
  1. Linear projections `Q, K, V`: `[2, 6, 3] -> [2, 6, 2]`
  2. Unroll heads (`.view`): `[2, 6, 2] -> [2, 6, 2, 1]` (`[B, T, num_heads, head_dim]`)
  3. Swap token & head dims (`.transpose(1, 2)`): `[2, 6, 2, 1] -> [2, 2, 6, 1]` (`[B, num_heads, T, head_dim]`)
  4. Batched scores (`Q @ K.transpose(2, 3)`): `[2, 2, 6, 1] @ [2, 2, 1, 6] -> [2, 2, 6, 6]`
  5. Apply causal mask (`col > row` -> `-Infinity`), scale by `1.0 / Math.sqrt(head_dim)`, apply `softmax(dim=-1)`.
  6. Context vectors (`attn_weights @ V`): `[2, 2, 6, 6] @ [2, 2, 6, 1] -> [2, 2, 6, 1]`
  7. Swap back (`.transpose(1, 2)`): `[2, 2, 6, 1] -> [2, 6, 2, 1]`
  8. Merge heads (`.contiguous().view(B, T, d_out)`): `[2, 6, 2, 1] -> [2, 6, 2]`
  9. Final output projection (`out_proj`): `[2, 6, 2] -> [2, 6, 2]`

---

## 3. Ground-Truth Test Fixtures & Verification Data
*(Use these exact floats in JUnit tests to avoid Python vs. Java PRNG seed differences)*

### A. Input Sequence (`shape = [6, 3]`)
Tokens: `"Your journey starts with one step"`
```java
float[] INPUTS_FLAT = {
    0.43f, 0.15f, 0.89f, // Your     (x^1)
    0.55f, 0.87f, 0.66f, // journey  (x^2)
    0.57f, 0.85f, 0.64f, // starts   (x^3)
    0.22f, 0.58f, 0.33f, // with     (x^4)
    0.77f, 0.25f, 0.10f, // one      (x^5)
    0.05f, 0.80f, 0.55f  // step     (x^6)
};

```

### B. Exact PyTorch `torch.manual_seed(123)` Weights (`d_in = 3, d_out = 2`)

In Section 3.4, `torch.rand(3, 2)` with seed `123` produces these exact matrices:

```java
// W_query (shape [3, 2])
float[] W_QUERY_SEED_123 = {
    0.2961f, 0.5166f,
    0.2517f, 0.6886f,
    0.0740f, 0.8665f
};

// W_key (shape [3, 2])
float[] W_KEY_SEED_123 = {
    0.1366f, 0.1025f,
    0.1841f, 0.7264f,
    0.3153f, 0.6871f
};

// W_value (shape [3, 2])
float[] W_VALUE_SEED_123 = {
    0.0756f, 0.1966f,
    0.3164f, 0.4017f,
    0.1186f, 0.8274f
};

```

### C. Expected Intermediate & Final Outputs (Section 3.4, Unmasked)

Using `INPUTS_FLAT` and the `SEED_123` weights above (with `1e-4f` delta tolerance):

* **`query_2`** (token index 1, `"journey"`): `[0.4306f, 1.4551f]`
* **`keys`** (full `[6, 2]` matrix):
  `[[0.3669f, 0.7646f], [0.4433f, 1.1419f], [0.4361f, 1.1156f], [0.2408f, 0.6706f], [0.1827f, 0.3292f], [0.3275f, 0.9642f]]`
* **`attn_scores_2`** (unscaled dot product `query_2 @ keys.T`):
  `[1.2705f, 1.8524f, 1.8111f, 1.0795f, 0.5577f, 1.5440f]`
* **`attn_weights_2`** (`softmax(attn_scores_2 / sqrt(2))`):
  `[0.1500f, 0.2264f, 0.2199f, 0.1311f, 0.0906f, 0.1820f]`
* **`context_vec_2`** (`attn_weights_2 @ values`):
  `[0.3061f, 0.8210f]`
* **Full `SelfAttention_v1` Output (`shape = [6, 2]`)**:
```java
float[][] EXPECTED_SA_V1 = {
    {0.2996f, 0.8053f},
    {0.3061f, 0.8210f},
    {0.3058f, 0.8203f},
    {0.2948f, 0.7939f},
    {0.2927f, 0.7891f},
    {0.2990f, 0.8040f}
};

```



### D. Reference Target Code (PyTorch Listing 3.5 — Section 3.6.2)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

```

---

## How to Begin

When I reference this file, acknowledge the rules, do NOT write any code, and ask me the **Milestone 1, Step A (Top-Down)** Socratic questions to design our `Tensor` record and its stride invariants.

```

<FollowUp label="Want a minimal JUnit 5 test template pre-loaded with these exact arrays?" query="Generate a minimal JUnit 5 test file containing those exact float arrays and assertions (with placeholders for my Tensor/Attention calls) so I can drop it straight into src/test/java."/>

```