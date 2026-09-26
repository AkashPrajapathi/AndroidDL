package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.op.ReLUOp

class ReLU : Module() {
    override fun invoke(input: Tensor): Tensor {
        return ReLUOp.apply(input)
    }
}