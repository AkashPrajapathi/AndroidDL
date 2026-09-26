package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

class ExpBackward(
    private val output: Tensor,
    input: Tensor
) : GradFn {

    override val nextEdges =
        listOf(Edge(input))

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val gradInput =
            gradOutput * output

        return listOf(gradInput)
    }
}