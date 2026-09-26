package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.sumToShape
import io.github.akashprajapathi.androiddl.core.transposeLastTwo

class MatMulBackward(
    left: Tensor,
    right: Tensor
) : GradFn {

    override val nextEdges =
        listOf(
            Edge(left),
            Edge(right)
        )

    override fun backward(
        gradOutput: Tensor
    ): List<Tensor?> {

        val left =
            nextEdges[0].tensor

        val right =
            nextEdges[1].tensor

        val gradLeft =
            gradOutput
                .matmul(
                    right.transposeLastTwo()
                )
                .sumToShape(left.shape)

        val gradRight =
            left
                .transposeLastTwo()
                .matmul(gradOutput)
                .sumToShape(right.shape)

        return listOf(
            gradLeft,
            gradRight
        )
    }
}