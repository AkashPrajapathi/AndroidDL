package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.expandReducedGradient

class MeanBackward(
    input: Tensor,
    private val axes: IntArray,
    private val keepDim: Boolean
) : GradFn {

    override val nextEdges =
        listOf(Edge(input))

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val input =
            nextEdges[0].tensor

        var reductionCount = 1

        for (axis in axes) {
            reductionCount *= input.shape[axis]
        }

        val gradInput =
            gradOutput
                .expandReducedGradient(
                    inputShape = input.shape,
                    axes = axes,
                    keepDim = keepDim
                )/ Tensor(data = floatArrayOf(reductionCount.toFloat()), shape = intArrayOf(1))

        return listOf(gradInput)
    }
}