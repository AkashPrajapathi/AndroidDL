package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.sumToShape

class SubtractBackward(
    a: Tensor,
    b: Tensor
) : GradFn {

    override val nextEdges = listOf<Edge>(
        Edge(a),
        Edge(b)
    )

    override fun backward(gradOutput: Tensor): List<Tensor?> {
        val gradA = gradOutput.sumToShape(nextEdges[0].tensor.shape)
        val gradB = gradOutput.sumToShape(nextEdges[1].tensor.shape)

        return listOf(
            gradA,
            -gradB
        )
    }
}