package com.galt.llm;

import java.util.Arrays;

public class LayerNorm {


    //mean & std provide the safety (stops numbers from spiraling out of control).
    //scale & shift provide the freedom (lets the AI adapt the scale of each feature dimension as needed).

    // initialized to all 1.0f
    private final Tensor scale;
    // initialized to all 0.0f
    private final Tensor shift;
    private final float eps;
    private final int embDim;

    // Default constructor: initializes scale to 1.0f, shift to 0.0f, eps to 1e-5f
    public LayerNorm(int embDim) {
        float[] s = new float[embDim];
        Arrays.fill(s, 1.0f);
        this.scale = Tensor.of(s, embDim);

        float[] b = new float[embDim];
        Arrays.fill(b, 0.0f);
        this.shift = Tensor.of(b, embDim);

        this.eps = 1e-5f;
        this.embDim = embDim;
    }

    public LayerNorm(Tensor scale, Tensor shift) {
        this.scale = scale;
        this.shift = shift;
        this.embDim = scale.shape()[0];
        this.eps = 1e-5f;
    }

    public Tensor forward(Tensor x) {
        int rank = x.shape().length;
        if (x.shape()[rank - 1] != embDim) {
            throw new IllegalArgumentException(
                "Last dimension " + x.shape()[rank - 1] + " does not match embDim " + embDim
            );
        }

        Tensor result = Tensor.zeros(x.shape());

        if (rank == 2) {
            int T = x.shape()[0];
            for (int t = 0; t < T; t++) {
                // 1. Mean
                float mean = 0.0f;
                for (int i = 0; i < embDim; i++) {
                    mean += x.get(t, i);
                }
                mean /= embDim;

                // 2. Biased Variance
                float variance = 0.0f;
                for (int i = 0; i < embDim; i++) {
                    float diff = x.get(t, i) - mean;
                    variance += diff * diff;
                }
                variance /= embDim;

                // 3. Normalize, Scale, and Shift
                float std = (float) Math.sqrt(variance + eps);
                for (int i = 0; i < embDim; i++) {
                    float normalized = (x.get(t, i) - mean) / std;
                    float val = (scale.get(i) * normalized) + shift.get(i);
                    result.set(val, t, i);
                }
            }
        } else if (rank == 3) {
            int B = x.shape()[0];
            int T = x.shape()[1];
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    // 1. Mean
                    float mean = 0.0f;
                    for (int i = 0; i < embDim; i++) {
                        mean += x.get(b, t, i);
                    }
                    mean /= embDim;

                    // 2. Biased Variance
                    float variance = 0.0f;
                    for (int i = 0; i < embDim; i++) {
                        float diff = x.get(b, t, i) - mean;
                        variance += diff * diff;
                    }
                    variance /= embDim;

                    // 3. Normalize, Scale, and Shift
                    float std = (float) Math.sqrt(variance + eps);
                    for (int i = 0; i < embDim; i++) {
                        float normalized = (x.get(b, t, i) - mean) / std;
                        float val = (scale.get(i) * normalized) + shift.get(i);
                        result.set(val, b, t, i);
                    }
                }
            }
        } else {
            throw new UnsupportedOperationException("LayerNorm only supports 2D and 3D tensors, got rank: " + rank);
        }

        return result;
    }
}
