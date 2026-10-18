package com.galt.llm;

public class FeedForward {
    private final Tensor w1; //shape : [emb_dim, 4* embdim]
    private final Tensor b1; // shape: [4 * emb_dim]
    private final Tensor w2;
    private final Tensor b2;
    private final GELU gelu;

    public FeedForward(Tensor w1, Tensor b1, Tensor w2, Tensor b2) {
        this.w1 = w1;
        this.b1 = b1;
        this.w2 = w2;
        this.b2 = b2;
        this.gelu = new GELU();
    }

    public Tensor forward(Tensor x){
        //Expand
        Tensor h1 = x.matmul(w1).addBias(b1);
        //Activate
        Tensor h2 = gelu.forward(h1);
        Tensor out = h2.matmul(w2).addBias(b2);

        return out;
    }
}
