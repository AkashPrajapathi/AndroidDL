package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor

class Parameter (
    val value: Tensor
){
    init {
        value.requiresGrad = true
    }
}