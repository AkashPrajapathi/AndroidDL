package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor
import kotlin.math.sqrt
import kotlin.random.Random

class Linear(
    val inFeatures: Int,
    val outFeatures: Int,
    val useBias: Boolean = true
) : Module() {

    // Xavier/Glorot uniform distribution scaling factor
    private val limit = sqrt(6.0 / (inFeatures + outFeatures)).toFloat()

    val weight: Parameter = registerParameter(
        "weight",
        Parameter(
            Tensor(
                data = FloatArray(inFeatures * outFeatures) {
                    Random.nextDouble(-limit.toDouble(), limit.toDouble()).toFloat()
                },
                shape = intArrayOf(inFeatures, outFeatures)
            )
        )
    )

    val bias: Parameter? = if (useBias) {
        registerParameter(
            "bias",
            Parameter(
                Tensor(
                    data = FloatArray(outFeatures) { 0f },
                    shape = intArrayOf(1, outFeatures)
                )
            )
        )
    } else null


    override fun invoke(input: Tensor): Tensor {
        var output = input.matmul(weight.value)
        bias?.let {
            output += it.value // Uses AddOp with broadcasting
        }
        return output
    }

}