package io.github.akashprajapathi.tic_tac_toeai.rl

import io.github.akashprajapathi.androiddl.core.Tensor

data class PolicyAction(
    val action: Int,
    val logProb: Tensor,
    val value: Tensor
)