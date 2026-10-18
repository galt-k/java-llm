package com.galt.llm.config;

public record GPTConfig(
        int vocabSize,
        int contextLength,
        int embDim,
        int nHeads,
        int nLayers,
        boolean qkvBias
) {
    //factory for standard GPT 2
    public static GPTConfig gpt2_124M(){
        return new GPTConfig(50257,
                1024,
                768,
                12,
                12,
                false
                );
    }
}
