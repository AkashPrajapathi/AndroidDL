package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.sumToShape

class MulBackward(
    a: Tensor,
    b: Tensor,
) : GradFn {
    override val nextEdges = listOf<Edge>(
        Edge(a),
        Edge(b),
    )

    override fun backward(gradOutput: Tensor): List<Tensor?> {
        val gradA = gradOutput * nextEdges[1].tensor
        val gradB = gradOutput * nextEdges[0].tensor
        return listOf(
            gradA.sumToShape(nextEdges[0].tensor.shape),
            gradB.sumToShape(nextEdges[1].tensor.shape),
        )
    }
}