package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.expandReducedGradient

class MinBackward(
    input: Tensor,
    private val winnerMask: Tensor,
    private val axes: IntArray,
    private val keepDim: Boolean
) : GradFn {

    override val nextEdges =
        listOf(
            Edge(input)
        )

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val input =
            nextEdges[0].tensor

        val expandedGradient =
            gradOutput.expandReducedGradient(
                inputShape = input.shape,
                axes = axes,
                keepDim = keepDim
            )

        val gradInput =
            expandedGradient * winnerMask

        return listOf(gradInput)
    }
}