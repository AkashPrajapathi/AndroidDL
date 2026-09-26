package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.ReLUBackward
import kotlin.math.max

object ReLUOp {
    fun apply(input: Tensor): Tensor {
        val result = FloatArray(input.numel)
        val inData = input.data

        for (i in inData.indices) {
            result[i] = max(0f, inData[i])
        }

        val output = Tensor(result, input.shape.copyOf())

        if (input.requiresGrad) {
            output.requiresGrad = true
            output.gradFn = ReLUBackward(input)
        }

        return output
    }
}