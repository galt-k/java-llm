package com.galt.llm;

public class TransformerBlock {
    private final MultiHeadAttention att;
    private final FeedForward ff;
    private final LayerNorm norm1;
    private final LayerNorm norm2;

    public TransformerBlock(MultiHeadAttention multiHeadAttention,
                            FeedForward feedForward,
                            LayerNorm norm1,
                            LayerNorm norm2){
        this.att = multiHeadAttention;
        this.ff = feedForward;
        this.norm1 = norm1;
        this.norm2 = norm2;
    }

    public Tensor forward(Tensor x, boolean causalMask){
        //Attention +shortcut
        Tensor shortcut = x;
        x = norm1.forward(x);
        x = att.forward(x, causalMask);
        x = shortcut.add(x);

        //feedforward + shortcut
        shortcut = x;
        x = norm2.forward(x);
        x = ff.forward(x);
        x= shortcut.add(x);

        return x;
    }
}
