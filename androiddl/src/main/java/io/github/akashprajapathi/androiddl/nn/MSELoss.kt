package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor

class MSELoss : Module() {

    override fun invoke(input: Tensor): Tensor {
        throw UnsupportedOperationException("MSELoss requires both prediction and target tensors.")
    }

    // Overload to accept predictions and targets
    operator fun invoke(pred: Tensor, target: Tensor): Tensor {
        val diff = pred - target        // Uses SubtractOp
        val squared = diff * diff       // Uses MulOp
        // Mean over all elements to return a scalar tensor [1]
        return squared.mean(
            axes = IntArray(squared.shape.size) { it },
            keepDim = false
        )
    }
}