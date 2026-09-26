package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

class ReshapeBackward(
    input: Tensor
) : GradFn {

    override val nextEdges =
        listOf(Edge(input))

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val input = nextEdges[0].tensor

        val gradInput =
            gradOutput.reshape(*input.shape)

        return listOf(gradInput)
    }
}