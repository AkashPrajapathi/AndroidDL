package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.ExpBackward

object ExpOp {

    fun apply(input: Tensor): Tensor {

        val result =
            FloatArray(input.numel)

        var i = 0

        for (value in input) {
            result[i++] = kotlin.math.exp(value.toDouble()).toFloat()
        }

        val output =
            Tensor(
                data = result,
                shape = input.shape.copyOf(),
                requiresGrad = input.requiresGrad
            )

        if (input.requiresGrad) {
            output.gradFn =
                ExpBackward(
                    input = input,
                    output = output
                )
        }

        return output
    }
}