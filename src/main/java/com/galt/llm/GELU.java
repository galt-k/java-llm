package com.galt.llm;

public class GELU {
    //Stateless module that has a forward method.
    public Tensor forward(Tensor x){
        Tensor c = x.contiguous();
        float[] out = new float[c.data().length];
        /*
        Iterate over each eleemgnt of c.data
         */
        int index = 0;
        for(float val: c.data()) {
            float inner = (float) Math.sqrt(2.0 / Math.PI) * (val + 0.044715f * val * val * val);
            float geluVal = 0.5f * val * (1.0f + (float) Math.tanh(inner));

            //Store geluVal intot out[i];
            out[index] = geluVal;
            index++;
        }

        return new Tensor(out,c.shape(),c.strides(),c.offset());
    }
}
