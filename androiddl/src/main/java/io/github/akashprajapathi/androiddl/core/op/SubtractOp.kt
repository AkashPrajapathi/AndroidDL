package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.SubtractBackward
import io.github.akashprajapathi.androiddl.core.elementwise

object SubtractOp {
    fun apply(a: Tensor, b: Tensor): Tensor {
        val output = elementwise(a, b) { x, y ->
            x - y
        }

        if (a.requiresGrad || b.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = SubtractBackward(a, b)
        }

        return output
    }
}