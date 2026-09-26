package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.MinBackward
import io.github.akashprajapathi.androiddl.core.computeOutputShape
import io.github.akashprajapathi.androiddl.core.iterator.ReductionIterator
import io.github.akashprajapathi.androiddl.core.outputIndex
import io.github.akashprajapathi.androiddl.core.reductionInputIndex

object MinOp {

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
            FloatArray(outputSize) {
                Float.POSITIVE_INFINITY
            }

        val winnerMask =
            FloatArray(input.numel)

        val winnerIndices =
            IntArray(outputSize) { -1 }

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

            val outputIndex =
                outputIndex(
                    position.outputCoordinate,
                    outputShape
                )

            if (value < result[outputIndex]) {

                val previousWinner =
                    winnerIndices[outputIndex]

                if (previousWinner != -1) {
                    winnerMask[previousWinner] = 0f
                }

                result[outputIndex] = value

                val logicalInputIndex =
                    reductionInputIndex(
                        outputCoordinate =
                            position.outputCoordinate,
                        reductionCoordinates =
                            position.reductionCoordinates,
                        axes = axes,
                        inputShape = input.shape
                    )

                winnerIndices[outputIndex] =
                    logicalInputIndex

                winnerMask[logicalInputIndex] = 1f
            }
        }

        val output =
            Tensor(
                data = result,
                shape = outputShape,
                requiresGrad = input.requiresGrad
            )

        if (input.requiresGrad) {

            output.gradFn =
                MinBackward(
                    input = input,
                    winnerMask =
                        Tensor(
                            data = winnerMask,
                            shape = input.shape.copyOf()
                        ),
                    axes = axes.copyOf(),
                    keepDim = keepDim
                )
        }

        return output
    }
}