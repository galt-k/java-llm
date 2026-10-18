package com.galt.llm;

public class SelfAttention {
    private final Tensor wQuery;
    private final Tensor wKey;
    private final Tensor wValue;
    private final int dout;

    public SelfAttention(Tensor wQuery,
                          Tensor wKey,
                          Tensor wValue
    ){
        this.wQuery = wQuery;
        this.wKey = wKey;
        this.wValue = wValue;
        this.dout = wQuery.shape()[1];

    }

    public Tensor forward(Tensor x, boolean causalMask){
        //Project the Query vector for the input token
        Tensor Q = x.matmul(wQuery);
        //Project the Key vector for the input token
        Tensor K = x.matmul(wKey);
        //Project the value vector for the input
        Tensor V = x.matmul(wValue);

        //Compute the Raw scores
        Tensor scores = Q.matmul(K.transpose(0,1));

        //scale
        Tensor scaledScores = scores.scale(1.0f / (float) Math.sqrt(this.dout));

        //causal mask
        if(causalMask){
            int T = scaledScores.shape()[0];
            for(int r = 0; r < T; r++){
                for(int c = 0; c < T; c++){
                    if(c > r){
                        scaledScores.set(Float.NEGATIVE_INFINITY, r, c);
                    }
                }
            }
        }

        //Softmax
        Tensor weights = scaledScores.softmax(-1);
        Tensor context = weights.matmul(V);
        return context;
    }
}
