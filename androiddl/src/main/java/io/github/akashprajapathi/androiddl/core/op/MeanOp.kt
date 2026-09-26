package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.MeanBackward
import io.github.akashprajapathi.androiddl.core.computeOutputShape
import io.github.akashprajapathi.androiddl.core.iterator.ReductionIterator
import io.github.akashprajapathi.androiddl.core.outputIndex

object MeanOp {

    fun apply(
        input: Tensor,
        axes: IntArray,
        keepDim: Boolean = false
    ): Tensor {

        for (axis in axes) {
            require(axis in input.shape.indices) {
                "Invalid axis $axis for tensor of rank ${input.shape.size}."
            }
        }

        val outputShape =
            computeOutputShape(
                input.shape,
                axes,
                keepDim
            )

        val outputSize =
            outputShape.fold(1) { acc, dim ->
                acc * dim
            }

        val result =
            FloatArray(outputSize)

        val iterator =
            ReductionIterator(
                shape = input.shape,
                strides = input.strides,
                offset = input.offset,
                axes = axes,
                keepDim = keepDim
            )

        while (iterator.hasNext()) {

            val position = iterator.next()

            val value =
                input.data[position.physicalIndex]

            val index =
                outputIndex(
                    position.outputCoordinate,
                    outputShape
                )

            result[index] += value
        }

        var reductionCount = 1

        for (axis in axes) {
            reductionCount *= input.shape[axis]
        }

        for (i in result.indices) {
            result[i] /= reductionCount
        }

        val output =
            Tensor(
                data = result,
                shape = outputShape,
                requiresGrad = input.requiresGrad
            )

        if (input.requiresGrad) {
            output.gradFn =
                MeanBackward(
                    input = input,
                    axes = axes.copyOf(),
                    keepDim = keepDim
                )
        }

        return output
    }
}