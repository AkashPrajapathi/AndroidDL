package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.expandReducedGradient

class SumBackward(
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

        val gradInput =
            gradOutput.expandReducedGradient(
                inputShape = input.shape,
                axes = axes,
                keepDim = keepDim
            )

        return listOf(gradInput)
    }
}