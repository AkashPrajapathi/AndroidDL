package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

class NegBackward(
    private val a: Tensor,
) : GradFn {
    override val nextEdges = listOf<Edge>(
        Edge(a),
    )

    override fun backward(gradOutput: Tensor): List<Tensor?> {
        return listOf(
            -gradOutput
        )
    }
}