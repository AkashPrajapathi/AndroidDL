package io.github.akashprajapathi.androiddl.optim

import io.github.akashprajapathi.androiddl.core.Tensor

abstract class Optimizer(
    protected val parameters: List<Tensor>
) {
    // Computes and applies weight updates in-place
    abstract fun step()

    // Resets gradients of all parameters before the next iteration
    fun zeroGrad() {
        for (param in parameters) {
            param.grad = null
        }
    }

    // Export optimizer internal states (e.g. momentum vectors)
    open fun stateDict(): Map<String, FloatArray> = emptyMap()

    // Load optimizer internal states in-place
    open fun loadStateDict(stateDict: Map<String, FloatArray>) {}
}