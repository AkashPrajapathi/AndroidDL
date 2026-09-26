package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.MulBackward
import io.github.akashprajapathi.androiddl.core.elementwise

object MulOp {
    fun apply(a: Tensor, b: Tensor): Tensor {
        val output = elementwise(a, b) { x, y -> x * y }

        if (a.requiresGrad || b.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = MulBackward(a, b)
        }

        return output
    }
}