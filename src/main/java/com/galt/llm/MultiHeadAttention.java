package com.galt.llm;

public class MultiHeadAttention {
    private final Tensor wQuery;
    private final Tensor bQuery;
    private final Tensor wKey;
    private final Tensor bKey;
    private final Tensor wValue;
    private final Tensor bValue;
    private final Tensor outProj;
    private final Tensor bOutProj;
    private final int numHeads;
    private final int headDim;
    private final int dOut;

    public MultiHeadAttention(Tensor wQuery,
                              Tensor wKey,
                              Tensor wValue,
                              Tensor outProj,
                              int numHeads) {
        this(wQuery, null, wKey, null, wValue, null, outProj, null, numHeads);
    }

    public MultiHeadAttention(Tensor wQuery, Tensor bQuery,
                              Tensor wKey, Tensor bKey,
                              Tensor wValue, Tensor bValue,
                              Tensor outProj, Tensor bOutProj,
                              int numHeads) {
        this.wQuery = wQuery;
        this.bQuery = bQuery;
        this.wKey = wKey;
        this.bKey = bKey;
        this.wValue = wValue;
        this.bValue = bValue;
        this.outProj = outProj;
        this.bOutProj = bOutProj;
        this.numHeads = numHeads;
        this.dOut = wQuery.shape()[1];

        if (dOut % numHeads != 0) {
            throw new IllegalArgumentException("dOut must be divisible by numHeads");
        }
        this.headDim = dOut / numHeads;
    }

    public Tensor forward(Tensor x, boolean causalMask) {
        int B = x.shape()[0];
        int T = x.shape()[1];

        // Step 1: Project [B, T, d_in] @ [d_in, d_out] -> [B, T, d_out]
        Tensor Q = x.matmul(wQuery);
        if (bQuery != null) {
            Q = Q.addBias(bQuery);
        }
        Tensor K = x.matmul(wKey);
        if (bKey != null) {
            K = K.addBias(bKey);
        }
        Tensor V = x.matmul(wValue);
        if (bValue != null) {
            V = V.addBias(bValue);
        }

        // Step 2: Unroll heads -> [B, T, numHeads, headDim]
        Q = Q.view(B, T, numHeads, headDim);
        K = K.view(B, T, numHeads, headDim);
        V = V.view(B, T, numHeads, headDim);

        // Step 3: Swap tokens & heads -> [B, numHeads, T, headDim]
        Q = Q.transpose(1, 2);
        K = K.transpose(1, 2);
        V = V.transpose(1, 2);

        // Step 4: Batched attention scores [B, H, T, headDim] @ [B, H, headDim, T] -> [B, H, T, T]
        Tensor scores = Q.matmul(K.transpose(2, 3));

        // Step 5: Scale by sqrt(headDim)
        Tensor scaledScores = scores.scale(1.0f / (float) Math.sqrt(headDim));

        // Step 5b: Causal mask (mask future tokens where c > r)
        if (causalMask) {
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int r = 0; r < T; r++) {
                        for (int c = 0; c < T; c++) {
                            if (c > r) {
                                scaledScores.set(Float.NEGATIVE_INFINITY, b, h, r, c);
                            }
                        }
                    }
                }
            }
        }

        // Softmax across the last dimension -> [B, H, T, T]
        Tensor weights = scaledScores.softmax(-1);

        // Step 6: Context vectors [B, H, T, T] @ [B, H, T, headDim] -> [B, H, T, headDim]
        Tensor context = weights.matmul(V);

        // Step 7: Swap back -> [B, T, numHeads, headDim]
        context = context.transpose(1, 2);

        // Step 8: Merge heads -> [B, T, dOut]
        context = context.contiguous().view(B, T, dOut);

        // Step 9: Final output projection [B, T, dOut] @ [dOut, dOut] -> [B, T, dOut]
        Tensor projected = context.matmul(outProj);
        if (bOutProj != null) {
            projected = projected.addBias(bOutProj);
        }
        return projected;
    }
}
