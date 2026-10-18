package com.galt.llm;

public record Tensor(float[] data,
                     int[] shape,
                     int[] strides,
                     int offset
                    )
{

    private static int[] computeContiguousStrides(int[] shape) {
        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        return strides;
    }

    /**
     * Translates multi-dimensional coordinates (indices) into the physical 1D array index.
     * Formula: offset + sum(indices[i] * strides[i])
     */
    public int flatIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "Expected " + shape.length + " indices, but got " + indices.length
            );
        }
        int idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    public float get(int... indices) {
        return data[flatIndex(indices)];
    }

    public void set(float value, int... indices) {
        data[flatIndex(indices)] = value;
    }

    public int rank() {
        return shape.length;
    }

    public static Tensor of(float[] data, int... shape) {
        int expected = 1;
        for (int s : shape) {
            expected *= s;
        }
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " does not match shape total " + expected
            );
        }
        int[] strides = computeContiguousStrides(shape);
        return new Tensor(data, shape, strides, 0);
    }

    public static Tensor zeros(int... shape) {
        int total = 1;
        for (int s : shape) {
            total *= s;
        }
        return of(new float[total], shape);
    }

    public Tensor transpose(int dim0, int dim1) {
        if (dim0 < 0 || dim0 >= shape.length || dim1 < 0 || dim1 >= shape.length) {
            throw new IllegalArgumentException(
                    "Invalid transpose dimensions: " + dim0 + ", " + dim1 + " for rank " + shape.length
            );
        }
        int[] newShape = shape.clone();
        int[] newStrides = strides.clone();

        int temp = newShape[dim0];
        newShape[dim0] = newShape[dim1];
        newShape[dim1] = temp;

        temp = newStrides[dim0];
        newStrides[dim0] = newStrides[dim1];
        newStrides[dim1] = temp;

        return new Tensor(data, newShape, newStrides, offset);
    }

    public boolean isContiguous() {
        int[] expected = computeContiguousStrides(shape);
        return java.util.Arrays.equals(strides, expected);
    }

    public Tensor view(int... newShape) {
        if (!isContiguous()) {
            throw new IllegalStateException("Tensor is not contiguous; call contiguous() before view()");
        }
        int currentTotal = 1;
        for (int s : shape) {
            currentTotal *= s;
        }
        int newTotal = 1;
        for (int s : newShape) {
            newTotal *= s;
        }
        if (currentTotal != newTotal) {
            throw new IllegalArgumentException(
                    "Cannot reshape tensor of size " + currentTotal + " into shape of size " + newTotal
            );
        }
        int[] newStrides = computeContiguousStrides(newShape);
        return new Tensor(data, newShape, newStrides, offset);
    }

    public Tensor contiguous() {
        int total = 1;
        for (int s : shape) {
            total *= s;
        }
        if (isContiguous() && offset == 0 && data.length == total) {
            return this;
        }

        float[] newData = new float[total];
        int[] coords = new int[shape.length];
        for (int i = 0; i < total; i++) {
            newData[i] = this.get(coords);
            for (int d = shape.length - 1; d >= 0; d--) {
                coords[d]++;
                if (coords[d] < shape[d]) {
                    break;
                }
                coords[d] = 0;
            }
        }
        return of(newData, shape);
    }

    /**
     * Matrix multiplication supporting 2D x 2D, 3D x 2D (batched input @ weight),
     * 3D x 3D, and 4D x 4D (batched multi-head attention).
     */
    public Tensor matmul(Tensor other) {
        int rA = this.shape.length;
        int rB = other.shape.length;

        if (rA < 2 || rB < 2) {
            throw new IllegalArgumentException("matmul requires at least 2D tensors, got ranks: " + rA + " and " + rB);
        }

        int kA = this.shape[rA - 1];
        int kB = other.shape[rB - 2];
        if (kA != kB) {
            throw new IllegalArgumentException(
                "Incompatible inner dimensions for matmul: " + kA + " != " + kB
            );
        }

        int M = this.shape[rA - 2];
        int N = other.shape[rB - 1];

        if (rA == 2 && rB == 2) {
            Tensor result = Tensor.zeros(M, N);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < kA; k++) {
                        sum += this.get(i, k) * other.get(k, j);
                    }
                    result.set(sum, i, j);
                }
            }
            return result;
        } else if (rA == 3 && rB == 2) {
            int B = this.shape[0];
            Tensor result = Tensor.zeros(B, M, N);
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kA; k++) {
                            sum += this.get(b, i, k) * other.get(k, j);
                        }
                        result.set(sum, b, i, j);
                    }
                }
            }
            return result;
        } else if (rA == 3 && rB == 3) {
            if (this.shape[0] != other.shape[0]) {
                throw new IllegalArgumentException("Batch dimensions must match: " + this.shape[0] + " != " + other.shape[0]);
            }
            int B = this.shape[0];
            Tensor result = Tensor.zeros(B, M, N);
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kA; k++) {
                            sum += this.get(b, i, k) * other.get(b, k, j);
                        }
                        result.set(sum, b, i, j);
                    }
                }
            }
            return result;
        } else if (rA == 4 && rB == 4) {
            if (this.shape[0] != other.shape[0] || this.shape[1] != other.shape[1]) {
                throw new IllegalArgumentException("Batch and head dimensions must match");
            }
            int B = this.shape[0];
            int H = this.shape[1];
            Tensor result = Tensor.zeros(B, H, M, N);
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < H; h++) {
                    for (int i = 0; i < M; i++) {
                        for (int j = 0; j < N; j++) {
                            float sum = 0.0f;
                            for (int k = 0; k < kA; k++) {
                                sum += this.get(b, h, i, k) * other.get(b, h, k, j);
                            }
                            result.set(sum, b, h, i, j);
                        }
                    }
                }
            }
            return result;
        } else {
            throw new UnsupportedOperationException("Unsupported ranks for matmul: " + rA + " and " + rB);
        }
    }

    /**
     * Numerically stable softmax along the specified dimension (defaulting to last axis).
     * Subtracts the row maximum before exponentiation to prevent IEEE 754 overflow.
     */
    public Tensor softmax(int dim) {
        int targetDim = (dim < 0) ? shape.length + dim : dim;
        if (targetDim != shape.length - 1) {
            throw new UnsupportedOperationException("Softmax currently implemented along the last dimension (dim = -1)");
        }

        Tensor result = Tensor.zeros(shape);
        int lastDim = shape.length - 1;
        int lastDimSize = shape[lastDim];

        if (shape.length == 1) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lastDimSize; i++) {
                max = Math.max(max, this.get(i));
            }
            float sum = 0.0f;
            for (int i = 0; i < lastDimSize; i++) {
                sum += (float) Math.exp(this.get(i) - max);
            }
            for (int i = 0; i < lastDimSize; i++) {
                float prob = (sum == 0.0f) ? 0.0f : (float) Math.exp(this.get(i) - max) / sum;
                result.set(prob, i);
            }
            return result;
        }

        int numRows = 1;
        for (int i = 0; i < lastDim; i++) {
            numRows *= shape[i];
        }

        int[] coords = new int[shape.length];
        for (int r = 0; r < numRows; r++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < lastDimSize; c++) {
                coords[lastDim] = c;
                max = Math.max(max, this.get(coords));
            }

            float sum = 0.0f;
            for (int c = 0; c < lastDimSize; c++) {
                coords[lastDim] = c;
                sum += (float) Math.exp(this.get(coords) - max);
            }

            for (int c = 0; c < lastDimSize; c++) {
                coords[lastDim] = c;
                float prob = (sum == 0.0f) ? 0.0f : (float) Math.exp(this.get(coords) - max) / sum;
                result.set(prob, coords);
            }

            for (int d = lastDim - 1; d >= 0; d--) {
                coords[d]++;
                if (coords[d] < shape[d]) {
                    break;
                }
                coords[d] = 0;
            }
        }

        return result;
    }

    public Tensor scale(float factor){
        //This should apply the formaula for every element in the resultant attention score tensor
        Tensor c = this.contiguous();

        float[] scaled = new float[c.data().length];
        for(int i = 0; i < c.data().length; i++) {
            scaled[i] = c.data()[i] * factor;
        }
        return new Tensor(scaled, c.shape().clone(), c.strides().clone(), 0);

    }

    public Tensor addBias(Tensor bias){
        //shape of bias = 4 * emb-dim
        // It's a 1D matrix
        // add it to every value in the tensor
        //we can just iterate throught the value and add it?
        Tensor c = this.contiguous();
        Tensor bContig = bias.contiguous();
        int D = bContig.data().length;

        ///Check the last dimension matches D
        if(c.shape()[c.shape().length - 1] != D) {
            throw new IllegalArgumentException("Bias dimension " + D + " does not match Tensor last dimension " + c.shape()[c.shape().length - 1]);
        }

        int index = 0;
        float[] out = new float[c.data().length];
        for(int i = 0; i <  c.data().length; i++){
            out[i] = c.data()[i] + bContig.data()[i % D];
        }

        //return a new Tensor
        return new Tensor(out, c.shape().clone(), c.strides().clone(), 0);
    }

    public Tensor add(Tensor other){
        if(!java.util.Arrays.equals(this.shape, other.shape())){
            throw new IllegalArgumentException("Shapes must match for addition");
        }

        Tensor a = this.contiguous();
        Tensor b = other.contiguous();

        //allocate output
        float[] out = new float[a.data().length];

        for(int i = 0; i < a.data().length; i++){
            out[i] = a.data()[i] + b.data()[i];
        }

        return new Tensor(out, a.shape().clone(), a.strides().clone(), 0);

    }

}
