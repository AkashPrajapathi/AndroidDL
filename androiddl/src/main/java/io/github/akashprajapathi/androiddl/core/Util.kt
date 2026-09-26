package io.github.akashprajapathi.androiddl.core

import io.github.akashprajapathi.androiddl.core.iterator.TensorIterator

fun outputIndex(
    coordinate: IntArray,
    shape: IntArray
): Int {

    var index = 0

    for (dimension in shape.indices) {
        index =
            index * shape[dimension] +
                    coordinate[dimension]
    }

    return index
}

fun computeOutputShape(
    shape: IntArray,
    axes: IntArray,
    keepDim: Boolean
): IntArray {

    if (keepDim) {

        val outputShape = shape.copyOf()

        for (axis in axes) {
            outputShape[axis] = 1
        }

        return outputShape
    }

    val outputShape =
        IntArray(shape.size - axes.size)

    var outputDimension = 0

    for (inputDimension in shape.indices) {

        if (inputDimension in axes) {
            continue
        }

        outputShape[outputDimension++] =
            shape[inputDimension]
    }

    return outputShape
}

fun broadcastShape(aShape: IntArray, bShape: IntArray): IntArray {
    val resultRank = maxOf(aShape.size, bShape.size)

    val resultShape = IntArray(resultRank)

    for (i in 0..<resultRank) {
        val aIndex = aShape.lastIndex - (resultRank - 1 - i)
        val bIndex = bShape.lastIndex - (resultRank - 1 - i)

        val aDim = if (aIndex >= 0) {
            aShape[aIndex]
        } else {
            1
        }

        val bDim = if (bIndex >= 0) {
            bShape[bIndex]
        } else {
            1
        }

        require(aDim == bDim || aDim == 1 || bDim == 1) {
            "Cannot broadcast shape " +
                    "${aShape.contentToString()} and " +
                    "${bShape.contentToString()}."
        }

        resultShape[i] = maxOf(aDim, bDim)
    }

    return resultShape
}

fun broadcastStrides(
    inputShape: IntArray,
    inputStrides: IntArray,
    outputShape: IntArray
): IntArray {
    val result =
        IntArray(outputShape.size)

    val rankDiff =
        outputShape.size - inputShape.size

    for (i in outputShape.indices) {

        if (i < rankDiff) {

            // Missing leading dimension.
            // This dimension is broadcast.
            result[i] = 0

            continue
        }

        val inputDimension =
            i - rankDiff

        result[i] =
            if (inputShape[inputDimension] == 1) {
                0
            } else {
                inputStrides[inputDimension]
            }
    }

    return result
}

fun broadcastIterator(
    tensor: Tensor,
    outputShape: IntArray
): Iterator<Float> {

    val effectiveStrides =
        broadcastStrides(
            inputShape = tensor.shape,
            inputStrides = tensor.strides,
            outputShape = outputShape
        )

    return TensorIterator(
        data = tensor.data,
        shape = outputShape,
        strides = effectiveStrides,
        offset = tensor.offset
    )
}

fun elementwise(
    a: Tensor,
    b: Tensor,
    operation: (Float, Float) -> Float
): Tensor {

    val outputShape = broadcastShape(a.shape, b.shape)

    val aIterator =
        broadcastIterator(
            tensor = a,
            outputShape = outputShape
        )

    val bIterator =
        broadcastIterator(
            tensor = b,
            outputShape = outputShape
        )

    val result = FloatArray(
        outputShape.fold(1) { acc, dim ->
            acc * dim
        }
    )

    var i = 0

    while (aIterator.hasNext()) {

        result[i++] =
            operation(
                aIterator.next(),
                bIterator.next()
            )
    }

    return Tensor(
        data = result,
        shape = outputShape
    )
}

fun Tensor.sumToShape(targetShape: IntArray): Tensor {

    // Scalar gradient → broadcast to target shape
    if (shape.isEmpty()) {
        val size = targetShape.fold(1) { acc, dim -> acc * dim }

        return Tensor(
            data = FloatArray(size) { data[0] },
            shape = targetShape.copyOf(),
            requiresGrad = false
        )
    }

    require(targetShape.size <= shape.size) {
        "Cannot reduce shape ${shape.contentToString()} " +
                "to ${targetShape.contentToString()}."
    }

    val leadingDimensions = shape.size - targetShape.size

    for (i in targetShape.indices) {
        val sourceDim = shape[leadingDimensions + i]
        val targetDim = targetShape[i]

        require(targetDim == sourceDim || targetDim == 1) {
            "Cannot reduce shape ${shape.contentToString()} " +
                    "to ${targetShape.contentToString()}."
        }
    }

    val axes = mutableListOf<Int>()

    for (i in 0..<leadingDimensions) {
        axes.add(i)
    }

    for (i in targetShape.indices) {
        val outputAxis = leadingDimensions + i

        if (targetShape[i] == 1 && shape[outputAxis] != 1) {
            axes.add(outputAxis)
        }
    }

    if (axes.isEmpty()) return this

    val reduced = this.sum(
        axes = axes.toIntArray(),
        keepDim = true
    )

    return reduced.reshape(*targetShape)
}

fun Tensor.expandReducedGradient(
    inputShape: IntArray,
    axes: IntArray,
    keepDim: Boolean
): Tensor {

    if (axes.isEmpty()) {
        return this
    }

    val reshapedShape =
        if (keepDim) {
            shape.copyOf()
        } else {
            inputShape.copyOf().also { shape ->
                for (axis in axes) {
                    shape[axis] = 1
                }
            }
        }

    val reshaped = reshape(*reshapedShape)

    val result =
        FloatArray(inputShape.fold(1) { acc, dim -> acc * dim })

    val iterator =
        broadcastIterator(
            tensor = reshaped,
            outputShape = inputShape
        )

    var i = 0

    while (iterator.hasNext()) {
        result[i++] = iterator.next()
    }

    return Tensor(
        data = result,
        shape = inputShape.copyOf()
    )
}

fun reductionInputIndex(
    outputCoordinate: IntArray,
    reductionCoordinates: IntArray,
    axes: IntArray,
    inputShape: IntArray
): Int {

    val coordinate = IntArray(inputShape.size)

    var outputIndex = 0
    var reductionIndex = 0

    for (dimension in inputShape.indices) {

        val reduced =
            dimension in axes

        if (reduced) {
            coordinate[dimension] =
                reductionCoordinates[reductionIndex++]
        } else {
            coordinate[dimension] =
                outputCoordinate[outputIndex++]
        }
    }

    return outputIndex(
        coordinate = coordinate,
        shape = inputShape
    )
}

fun Tensor.transposeLastTwo(): Tensor {

    require(shape.size >= 2) {
        "transposeLastTwo requires rank >= 2."
    }

    val dimensions =
        IntArray(shape.size) { it }

    val last =
        dimensions.lastIndex

    val secondLast =
        dimensions.lastIndex - 1

    dimensions[secondLast] = last
    dimensions[last] = secondLast

    return permute(*dimensions)
}

fun inversePermutation(
    dimensions: IntArray
): IntArray {

    val inverse =
        IntArray(dimensions.size)

    for (i in dimensions.indices) {
        inverse[dimensions[i]] = i
    }

    return inverse
}