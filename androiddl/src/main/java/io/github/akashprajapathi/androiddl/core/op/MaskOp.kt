package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.MaskBackward
import io.github.akashprajapathi.androiddl.core.broadcastIterator
import io.github.akashprajapathi.androiddl.core.broadcastShape

object MaskOp {

    fun apply(
        input: Tensor,
        mask: Tensor,
        value: Float
    ): Tensor {

        /*
         * Mask must broadcast to input.
         */
        val broadcastedShape =
            broadcastShape(
                input.shape,
                mask.shape
            )

        require(
            broadcastedShape.contentEquals(input.shape)
        ) {
            "Mask shape ${mask.shape.contentToString()} " +
                    "is not broadcastable to input shape " +
                    "${input.shape.contentToString()}."
        }

        val maskIterator =
            broadcastIterator(
                tensor = mask,
                outputShape = input.shape
            )

        val result =
            FloatArray(input.numel)

        var i = 0

        for (inputValue in input) {

            val maskValue =
                maskIterator.next()

            result[i++] =
                if (maskValue != 0f) {
                    inputValue
                } else {
                    value
                }
        }

        val output =
            Tensor(
                data = result,
                shape = input.shape.copyOf(),
                requiresGrad = input.requiresGrad
            )

        if (input.requiresGrad) {
            output.gradFn =
                MaskBackward(
                    input = input,
                    mask = mask
                )
        }

        return output
    }
}