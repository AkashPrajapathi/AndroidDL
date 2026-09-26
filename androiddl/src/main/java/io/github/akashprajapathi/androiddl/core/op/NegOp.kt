package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.NegBackward

object NegOp {

    fun apply(input: Tensor): Tensor {

        val result = FloatArray(input.numel)

        var i = 0

        for (value in input) {
            result[i++] = -value
        }

        val output = Tensor(
            data = result,
            shape = input.shape.copyOf(),
        )

        output.requiresGrad = input.requiresGrad

        if (input.requiresGrad) {
            output.gradFn = NegBackward(input)
        }

        return output
    }
}