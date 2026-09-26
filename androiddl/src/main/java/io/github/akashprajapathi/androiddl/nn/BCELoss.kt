package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor

class BCELoss(private val eps: Float = 1e-7f) : Module() {

    override fun invoke(input: Tensor): Tensor {
        throw UnsupportedOperationException("BCELoss requires predictions and target tensors.")
    }

    /**
     * @param predictions Probabilities (output of Sigmoid, values between 0 and 1)
     * @param targets Binary labels (0 or 1)
     */
    operator fun invoke(predictions: Tensor, targets: Tensor): Tensor {
        val ones = Tensor(FloatArray(predictions.numel) { 1f }, predictions.shape.copyOf())

        // Clamp probabilities using log to avoid log(0) = NaN
        val logPred = predictions.log()
        val logOneMinusPred = (ones - predictions).log()

        // Term 1: y * log(p)
        val term1 = targets * logPred

        // Term 2: (1 - y) * log(1 - p)
        val term2 = (ones - targets) * logOneMinusPred

        val bce = -(term1 + term2)

        return bce.mean(
            axes = IntArray(bce.shape.size) { it },
            keepDim = false
        )
    }
}