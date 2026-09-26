package io.github.akashprajapathi.tic_tac_toeai.rl

import io.github.akashprajapathi.androiddl.core.Tensor

class Episode {

    private data class Step(
        val logProb: Tensor,
        val value: Tensor,
        var reward: Float = 0f
    )

    private val steps = mutableListOf<Step>()

    fun add(
        logProb: Tensor,
        value: Tensor
    ) {
        steps += Step(
            logProb = logProb,
            value = value
        )
    }

    fun setLastReward(reward: Float) {
        require(steps.isNotEmpty()) {
            "Cannot set reward for an empty episode."
        }

        steps.last().reward = reward
    }

    fun size(): Int = steps.size

    fun isEmpty(): Boolean = steps.isEmpty()

    fun clear() {
        steps.clear()
    }

    /**
     * Computes n = 2 advantage estimates:
     *
     * A_t = δ_t + γ δ_{t+1}
     *
     * where
     *
     * δ_t = r_t + γ V(s_{t+1}) - V(s_t)
     */
    fun advantages(gamma: Float): FloatArray {

        val result = FloatArray(steps.size)

        for (t in steps.indices) {

            val current = steps[t]

            val v_t = current.value.data[0]

            // V(s_{t+1})
            val v_t1 =
                if (t + 1 < steps.size) {
                    steps[t + 1].value.data[0]
                } else {
                    0f
                }

            val delta_t =
                current.reward +
                        gamma * v_t1 -
                        v_t

            // δ_{t+1}
            val delta_t1 =
                if (t + 1 < steps.size) {

                    val next = steps[t + 1]

                    val v_t2 =
                        if (t + 2 < steps.size) {
                            steps[t + 2].value.data[0]
                        } else {
                            0f
                        }

                    next.reward +
                            gamma * v_t2 -
                            next.value.data[0]

                } else {
                    0f
                }

            result[t] =
                delta_t + gamma * delta_t1
        }

        return result
    }

    /**
     * Critic target corresponding to the n = 2 return:
     *
     * G_t^(2) =
     * r_t + γ r_{t+1} + γ² V(s_{t+2})
     */
    fun valueTargets(gamma: Float): FloatArray {

        val targets = FloatArray(steps.size)

        for (t in steps.indices) {

            val r_t = steps[t].reward

            val r_t1 =
                if (t + 1 < steps.size) {
                    steps[t + 1].reward
                } else {
                    0f
                }

            val v_t2 =
                if (t + 2 < steps.size) {
                    steps[t + 2].value.data[0]
                } else {
                    0f
                }

            targets[t] =
                r_t +
                        gamma * r_t1 +
                        gamma * gamma * v_t2
        }

        return targets
    }

    /**
     * Actor loss:
     *
     * L_actor = - A_t log π(a_t | s_t)
     */
    fun actorLoss(
        gamma: Float
    ): Tensor {

        require(steps.isNotEmpty()) {
            "Cannot calculate loss for an empty episode."
        }

        val advantages = advantages(gamma)

        var loss: Tensor? = null

        for (t in steps.indices) {

            val stepLoss =
                -steps[t].logProb * advantages[t]

            loss =
                if (loss == null) {
                    stepLoss
                } else {
                    loss + stepLoss
                }
        }

        return loss!!
    }

    /**
     * Critic loss:
     *
     * L_critic = (V(s_t) - G_t^(2))²
     */
    fun criticLoss(
        gamma: Float
    ): Tensor {

        require(steps.isNotEmpty()) {
            "Cannot calculate loss for an empty episode."
        }

        val targets = valueTargets(gamma)

        var loss: Tensor? = null

        for (t in steps.indices) {

            val difference =
                steps[t].value - targets[t]

            val stepLoss =
                difference * difference

            loss =
                if (loss == null) {
                    stepLoss
                } else {
                    loss + stepLoss
                }
        }

        return loss!!
    }
}