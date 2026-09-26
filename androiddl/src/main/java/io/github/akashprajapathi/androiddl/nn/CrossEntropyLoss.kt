package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor

class CrossEntropyLoss : Module(){

    override fun invoke(input: Tensor): Tensor {
        throw UnsupportedOperationException("CrossEntropyLoss requires predictions and one-hot target tensors.")
    }

    /**
     * @param predictions Logits (raw outputs before softmax) or Log-Softmax outputs of shape [BatchSize, NumClasses]
     * @param targets One-hot target labels of shape [BatchSize, NumClasses]
     * @param axis Dimension across which softmax is computed (default -1 or last axis)
     */

    operator fun invoke(predictions: Tensor, targets: Tensor, axis: Int = 1): Tensor {
        // If predictions are raw logits, compute logSoftmax first
        val logProbs = predictions.logSoftmax(axis = axis)

        // Elementwise multiplication of target * logProbs
        val product = targets * logProbs

        // Negative sum reduction over class axis, then mean over batch
        val sumLoss = -product.sum(axes = intArrayOf(axis), keepDim = true)

        return sumLoss.mean(
            axes = IntArray(sumLoss.shape.size) { it },
            keepDim = false
        )
    }
}