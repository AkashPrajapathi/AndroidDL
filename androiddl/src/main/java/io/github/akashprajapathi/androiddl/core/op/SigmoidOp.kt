package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.SigmoidBackward

object SigmoidOp {

    fun apply(input: Tensor): Tensor {
        val ones = Tensor(FloatArray(input.numel) { 1f }, input.shape.copyOf())

        // output = 1 / (1 + exp(-input))
        val output = ones / (ones + (-input).exp())

        if (input.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = SigmoidBackward(input, output)
        }

        return output
    }
}