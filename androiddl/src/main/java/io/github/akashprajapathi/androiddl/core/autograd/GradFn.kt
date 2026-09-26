package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

interface GradFn {

    val nextEdges: List<Edge>

    fun backward(
        gradOutput: Tensor
    ): List<Tensor?>
}