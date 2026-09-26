package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

class LogSoftmaxBackward(
    input: Tensor,
    private val output: Tensor,
    private val axis: Int
) : GradFn {

    override val nextEdges =
        listOf(Edge(input))

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val input = nextEdges[0].tensor

        val softmax =
            output.exp()

        val gradSum =
            gradOutput.sum(
                axes = intArrayOf(axis),
                keepDim = true
            )

        val gradInput =
            gradOutput -
                    softmax * gradSum

        return listOf(
            gradInput
        )
    }
}