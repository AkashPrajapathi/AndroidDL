package io.github.akashprajapathi.tic_tac_toeai.rl

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.nn.Linear
import io.github.akashprajapathi.androiddl.nn.Module
import io.github.akashprajapathi.androiddl.nn.ReLU
import io.github.akashprajapathi.androiddl.nn.Sequential

class PolicyNet : Module() {

    private val model = registerModule(
        "model",
        Sequential(
            Linear(inFeatures = 9, outFeatures = 32),
            ReLU(),
            Linear(inFeatures = 32, outFeatures = 32),
            ReLU(),
            Linear(inFeatures = 32, outFeatures = 9)
        )
    ) as Sequential

    override fun invoke(input: Tensor): Tensor {
        return model(input)
    }
}