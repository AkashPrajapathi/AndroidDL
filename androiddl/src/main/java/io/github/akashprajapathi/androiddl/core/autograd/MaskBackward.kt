package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.elementwise

class MaskBackward(
    input: Tensor,
    private val mask: Tensor
) : GradFn {

    override val nextEdges =
        listOf(Edge(input))

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val input = nextEdges[0].tensor

        val gradInput =
            elementwise(gradOutput, mask) { grad, m ->
                if (m != 0f) grad else 0f
            }

        return listOf(gradInput)
    }
}