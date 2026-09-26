package io.github.akashprajapathi.androiddl.optim

import io.github.akashprajapathi.androiddl.core.Tensor

class SGD(
    parameters: List<Tensor>,
    val lr: Float,
    val momentum: Float = 0f,
) : Optimizer(parameters) {

    // Internal state velocity map for momentum
    private val velocityMap = mutableMapOf<Tensor, FloatArray>()

    override fun step() {
        for (param in parameters) {
            val grad = param.grad ?: continue
            val data = param.data
            val gradData = grad.data

            if (momentum > 0f) {
                // Fetch or initialize velocity array using 'param' tensor
                val velocity = velocityMap.getOrPut(param) { FloatArray(param.numel) }

                for (i in data.indices) {
                    velocity[i] = momentum * velocity[i] + gradData[i]
                    data[i] -= lr * velocity[i]
                }
            } else {
                // Vanilla SGD
                for (i in data.indices) {
                    data[i] -= lr * gradData[i]
                }
            }
        }
    }

    // Serialize momentum states with index positions ("velocity_0", "velocity_1" etc.)
    override fun stateDict(): Map<String, FloatArray> {
        val dict = mutableMapOf<String, FloatArray>()
        if (momentum > 0f) {
            for ((index, param) in parameters.withIndex()) {
                val velocity = velocityMap[param]
                if (velocity != null) {
                    dict["velocity_$index"] = velocity.copyOf()
                }
            }
        }
        return dict
    }

    // Load velocity vectors back to parameters map
    override fun loadStateDict(stateDict: Map<String, FloatArray>) {
        if (momentum > 0f) {
            for ((index, param) in parameters.withIndex()) {
                val savedVelocity = stateDict["velocity_$index"] ?: continue
                require(savedVelocity.size == param.numel) {
                    "Velocity size mismatch for parameter index $index"
                }
                val velocity = velocityMap.getOrPut(param) { FloatArray(param.numel) }
                savedVelocity.copyInto(velocity)
            }
        }
    }
}