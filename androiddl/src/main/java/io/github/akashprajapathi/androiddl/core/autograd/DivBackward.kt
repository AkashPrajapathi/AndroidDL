package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.sumToShape

class DivBackward(
    a: Tensor,
    b: Tensor
) : GradFn {

    override val nextEdges =
        listOf(
            Edge(a),
            Edge(b)
        )

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val a = nextEdges[0].tensor
        val b = nextEdges[1].tensor

        val gradA =
            (gradOutput / b)
                .sumToShape(a.shape)

        val gradB =
            (-gradOutput * a / (b * b))
                .sumToShape(b.shape)

        return listOf(
            gradA,
            gradB
        )
    }
}