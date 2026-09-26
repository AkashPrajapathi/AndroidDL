package io.github.akashprajapathi.androiddl.core.iterator

import io.github.akashprajapathi.androiddl.core.computeOutputShape

class ReductionIterator(
    private val shape: IntArray,
    private val strides: IntArray,
    private val offset: Int,
    private val axes: IntArray,
    private val keepDim: Boolean = false
) : Iterator<ReductionPosition> {

    private val outputShape =
        computeOutputShape(shape, axes, keepDim)

    private val outputCoordinates =
        IntArray(outputShape.size)

    private var reductionCoordinates = IntArray(axes.size)
    private var finished = false

    override fun hasNext(): Boolean = !finished

    override fun next(): ReductionPosition {

        if (!hasNext()) {
            throw NoSuchElementException()
        }

        val inputCoordinate =
            mapOutputToInput(
                outputCoordinates,
                axes,
                reductionCoordinates,
                keepDim
            )

        var physicalIndex = offset

        for (dimension in shape.indices) {
            physicalIndex +=
                inputCoordinate[dimension] *
                        strides[dimension]
        }

        val position = ReductionPosition(
            outputCoordinate = outputCoordinates.copyOf(),
            reductionCoordinates = reductionCoordinates.copyOf(),
            physicalIndex = physicalIndex
        )

        advance()

        return position
    }

    private fun advanceReductionCoordinates(): Boolean {
        for (dim in reductionCoordinates.lastIndex downTo 0) {
            reductionCoordinates[dim]++

            val axis = axes[dim]

            if (reductionCoordinates[dim] < shape[axis]) {
                return true
            }

            reductionCoordinates[dim] = 0
        }

        return false
    }

    private fun advance() {

        // First move inside reduction space.
        if (advanceReductionCoordinates()) {
            return
        }

        reductionCoordinates.fill(0)

        for (dimension in outputCoordinates.lastIndex downTo 0) {

            if (keepDim && dimension in axes) {
                continue
            }

            outputCoordinates[dimension]++

            if (outputCoordinates[dimension] < outputShape[dimension]) {
                return
            }

            outputCoordinates[dimension] = 0
        }

        finished = true
    }

    private fun mapOutputToInput(
        outputCoordinate: IntArray,
        axes: IntArray,
        reductionCoordinates: IntArray,
        keepDim: Boolean
    ): IntArray {

        val inputCoordinate = IntArray(
            if (keepDim)
                outputCoordinate.size
            else
                outputCoordinate.size + axes.size
        )

        var outputDim = 0
        var reductionDim = 0

        for (inputDim in inputCoordinate.indices) {
            if (inputDim in axes) {
                inputCoordinate[inputDim] = reductionCoordinates[reductionDim++]
            } else {
                inputCoordinate[inputDim] = if (keepDim) {
                    outputCoordinate[inputDim]
                } else {
                    outputCoordinate[outputDim++]
                }
            }
        }

        return inputCoordinate
    }
}