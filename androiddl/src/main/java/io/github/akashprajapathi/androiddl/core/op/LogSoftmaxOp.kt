package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.LogSoftmaxBackward

object LogSoftmaxOp {

    fun apply(
        input: Tensor,
        axis: Int
    ): Tensor {

        require(input.shape.isNotEmpty()) {
            "LogSoftmax requires a tensor with at least one dimension."
        }

        val normalizedAxis =
            if (axis < 0) {
                axis + input.shape.size
            } else {
                axis
            }

        require(
            normalizedAxis in input.shape.indices
        ) {
            "Axis $axis is out of bounds for tensor of rank ${input.shape.size}."
        }

        /*
         * max along axis
         */
        val max =
            input.max(
                axes = intArrayOf(normalizedAxis),
                keepDim = true
            )

        /*
         * x - max
         */
        val shifted =
            input - max

        /*
         * exp(x - max)
         */
        val exp =
            shifted.exp()

        /*
         * sum(exp(x - max))
         */
        val sumExp =
            exp.sum(
                axes = intArrayOf(normalizedAxis),
                keepDim = true
            )

        /*
         * log(sum(exp(x - max)))
         */
        val logSumExp =
            sumExp.log()

        /*
         * logSoftmax
         *
         * x - max - log(sum(exp(x - max)))
         */
        val output =
            shifted - logSumExp

        if (input.requiresGrad) {

            output.gradFn =
                LogSoftmaxBackward(
                    input = input,
                    output = output,
                    axis = normalizedAxis
                )
        }

        return output
    }
}