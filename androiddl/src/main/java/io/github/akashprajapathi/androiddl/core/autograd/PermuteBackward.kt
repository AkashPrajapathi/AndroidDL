package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.inversePermutation

class PermuteBackward(
    input: Tensor,
    private val dimensions: IntArray
) : GradFn {

    override val nextEdges =
        listOf(Edge(input))

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val inverse =
            inversePermutation(dimensions)

        val gradInput =
            gradOutput.permute(*inverse)

        return listOf(gradInput)
    }
}