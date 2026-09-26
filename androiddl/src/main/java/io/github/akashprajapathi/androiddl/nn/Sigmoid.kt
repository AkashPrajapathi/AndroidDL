package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.op.SigmoidOp

class Sigmoid : Module() {
    override fun invoke(input: Tensor): Tensor {
        return SigmoidOp.apply(input)
    }
}