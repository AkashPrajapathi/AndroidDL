package io.github.akashprajapathi.tic_tac_toeai.rl

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.op.LogSoftmaxOp
import io.github.akashprajapathi.androiddl.core.op.MaskOp
import kotlin.random.Random

class PolicyAgent(
    private val policy: PolicyNet,
    private val valueNet: ValueNet
) {

    fun selectAction(states: IntArray): PolicyAction {

        require(states.size == 9) {
            "Tic-Tac-Toe state must contain 9 cells."
        }

        val stateTensor = Tensor(
            data = FloatArray(9) { index ->
                states[index].toFloat()
            },
            shape = intArrayOf(1, 9)
        )

        // Critic: V(s)
        val value = valueNet(stateTensor)

        // Actor: policy logits
        val logits = policy(stateTensor)

        val mask = Tensor(
            data = FloatArray(9) { index ->
                if (states[index] == 0) 1f else 0f
            },
            shape = intArrayOf(1, 9)
        )

        val maskedLogits = MaskOp.apply(
            input = logits,
            mask = mask,
            value = -Float.MAX_VALUE
        )

        val logProbs = LogSoftmaxOp.apply(
            input = maskedLogits,
            axis = 1
        )

        val probabilities = logProbs.exp()

        val action = sampleAction(
            probabilities = probabilities,
            legalMask = mask
        )

        val actionMask = Tensor(
            data = FloatArray(9) { index ->
                if (index == action) 1f else 0f
            },
            shape = intArrayOf(1, 9)
        )

        val selectedLogProb =
            (logProbs * actionMask).sum(
                axes = intArrayOf(1),
                keepDim = true
            )

        return PolicyAction(
            action = action,
            logProb = selectedLogProb,
            value = value
        )
    }

    private fun sampleAction(
        probabilities: Tensor,
        legalMask: Tensor
    ): Int {

        val randomValue = Random.nextFloat()

        var cumulative = 0f

        for (i in 0 until 9) {

            if (legalMask.data[i] == 0f) {
                continue
            }

            cumulative += probabilities.data[i]

            if (randomValue < cumulative) {
                return i
            }
        }

        for (i in 8 downTo 0) {
            if (legalMask.data[i] != 0f) {
                return i
            }
        }

        error("No legal actions available.")
    }
}