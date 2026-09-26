package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor

class Sequential(vararg layers: Module) : Module() {

    val layersList = layers.toList()

    init {
        for ((index, layer) in layersList.withIndex()) {
            registerModule("layer_$index", layer)
        }
    }

    override fun invoke(input: Tensor): Tensor {
        var current = input
        for (layer in layersList) {
            current = layer(current)
        }

        return current
    }
}