package io.github.akashprajapathi.androiddl.core

import io.github.akashprajapathi.androiddl.core.autograd.GradFn
import io.github.akashprajapathi.androiddl.core.autograd.PermuteBackward
import io.github.akashprajapathi.androiddl.core.autograd.ReshapeBackward
import io.github.akashprajapathi.androiddl.core.iterator.TensorIterator
import io.github.akashprajapathi.androiddl.core.op.AddOp
import io.github.akashprajapathi.androiddl.core.op.DivOp
import io.github.akashprajapathi.androiddl.core.op.ExpOp
import io.github.akashprajapathi.androiddl.core.op.LogOp
import io.github.akashprajapathi.androiddl.core.op.LogSoftmaxOp
import io.github.akashprajapathi.androiddl.core.op.MaskOp
import io.github.akashprajapathi.androiddl.core.op.MatMulOp
import io.github.akashprajapathi.androiddl.core.op.MaxOp
import io.github.akashprajapathi.androiddl.core.op.MeanOp
import io.github.akashprajapathi.androiddl.core.op.MinOp
import io.github.akashprajapathi.androiddl.core.op.MulOp
import io.github.akashprajapathi.androiddl.core.op.NegOp
import io.github.akashprajapathi.androiddl.core.op.ReLUOp
import io.github.akashprajapathi.androiddl.core.op.SigmoidOp
import io.github.akashprajapathi.androiddl.core.op.SubtractOp
import io.github.akashprajapathi.androiddl.core.op.SumOp

class Tensor private constructor(
    val data: FloatArray,
    val shape: IntArray,
    val strides: IntArray,
    val offset: Int,

    var requiresGrad: Boolean,

    internal var gradFn: GradFn? = null
) {

    val numel: Int
        get() = shape.fold(1) { acc, dim ->
            acc * dim
        }

    var grad: Tensor? = null
        internal set

    constructor(
        data: FloatArray,
        shape: IntArray,
        requiresGrad: Boolean = false
    ) : this(
        data = data,
        shape = shape,
        strides = calculateStrides(shape),
        offset = 0,
        requiresGrad = requiresGrad,
        gradFn = null
    ) {
        validateShapeAndData()
    }

    operator fun get(vararg indices: Int): Tensor {

        require(indices.size <= shape.size) {
            "Too many indices: ${indices.size} for " +
                    "tensor of rank ${shape.size}."
        }

        var newOffset = offset

        for (i in indices.indices) {

            require(indices[i] in 0 until shape[i]) {
                "Index ${indices[i]} out of bounds " +
                        "for dimension $i."
            }

            newOffset += indices[i] * strides[i]
        }

        val newShape =
            shape.copyOfRange(
                indices.size,
                shape.size
            )

        val newStrides =
            strides.copyOfRange(
                indices.size,
                strides.size
            )

        return createView(
            shape = newShape,
            strides = newStrides,
            offset = newOffset
        )
    }

    private fun createView(
        shape: IntArray,
        strides: IntArray,
        offset: Int,
        requiresGrad: Boolean = this.requiresGrad
    ): Tensor {
        return Tensor(
            data = data,
            shape = shape,
            strides = strides,
            offset = offset,
            requiresGrad = requiresGrad,
            gradFn = null
        )
    }

    companion object {

        private fun calculateStrides(
            shape: IntArray,
        ): IntArray {

            val strides = IntArray(shape.size)

            var stride = 1

            for (i in shape.lastIndex downTo 0) {
                strides[i] = stride
                stride *= shape[i]
            }

            return strides
        }
    }

    private fun validateShapeAndData() {

        require(shape.all { it >= 0 }) {
            "Shape dimensions must be non-negative."
        }

        val expectedSize = shape.fold(1) { acc, dim ->
            acc * dim
        }

        require(expectedSize == data.size) {
            "Data size ${data.size} does not match " +
                    "shape ${shape.contentToString()}."
        }
    }

    operator fun iterator(): Iterator<Float> {
        return TensorIterator(data, shape, strides, offset)
    }

    fun reshape(vararg newShape: Int): Tensor {

        require(newShape.all { it >= 0 }) {
            "Shape dimensions must be non-negative."
        }

        val newNumel =
            newShape.fold(1) { acc, dim ->
                acc * dim
            }

        require(newNumel == numel) {
            "Cannot reshape tensor of $numel elements " +
                    "into shape ${newShape.contentToString()}."
        }

        val result =
            if (isContiguous()) {

                createView(
                    shape = newShape.copyOf(),
                    strides = calculateStrides(newShape),
                    offset = offset
                )

            } else {

                val newStrides =
                    computeReshapeStrides(newShape)

                if (newStrides != null) {

                    createView(
                        shape = newShape.copyOf(),
                        strides = newStrides,
                        offset = offset
                    )

                } else {

                    val copiedData =
                        FloatArray(numel)

                    var i = 0

                    for (value in this) {
                        copiedData[i++] = value
                    }

                    Tensor(
                        data = copiedData,
                        shape = newShape.copyOf(),
                        requiresGrad = requiresGrad
                    )
                }
            }

        if (requiresGrad) {
            result.gradFn =
                ReshapeBackward(this)
        }

        return result
    }

    fun permute(vararg dimensions: Int): Tensor {

        require(dimensions.size == shape.size) {
            "Permutation must contain ${shape.size} dimensions."
        }

        require(dimensions.toSet().size == dimensions.size) {
            "Permutation contains duplicate dimensions."
        }

        require(dimensions.all { it in shape.indices }) {
            "Invalid dimension in permutation."
        }

        val newShape =
            IntArray(shape.size)

        val newStrides =
            IntArray(shape.size)

        for (i in dimensions.indices) {

            val oldDimension =
                dimensions[i]

            newShape[i] =
                shape[oldDimension]

            newStrides[i] =
                strides[oldDimension]
        }

        val result =
            createView(
                shape = newShape,
                strides = newStrides,
                offset = offset
            )

        if (requiresGrad) {
            result.gradFn =
                PermuteBackward(
                    input = this,
                    dimensions = dimensions.copyOf()
                )
        }

        return result
    }

    fun sum(
        axes: IntArray,
        keepDim: Boolean = false
    ): Tensor {
        return SumOp.apply(
            input = this,
            axes = axes,
            keepDim = keepDim
        )
    }

    fun mean(
        axes: IntArray,
        keepDim: Boolean = false
    ): Tensor {
        return MeanOp.apply(
            input = this,
            axes = axes,
            keepDim = keepDim
        )
    }

    fun min(
        axes: IntArray,
        keepDim: Boolean = false
    ): Tensor {
        return MinOp.apply(this, axes, keepDim)
    }

    fun max(
        axes: IntArray,
        keepDim: Boolean = false
    ): Tensor {
        return MaxOp.apply(this, axes, keepDim)
    }

    fun matmul(
        other: Tensor
    ): Tensor {
        return MatMulOp.apply(
            this,
            other
        )
    }

    operator fun plus(other: Tensor): Tensor {
        return AddOp.apply(this, other)
    }

    operator fun plus(other: Float): Tensor {
        return AddOp.apply(this, Tensor(data = floatArrayOf(other), shape = intArrayOf()))
    }

    operator fun times(other: Tensor): Tensor {
        return MulOp.apply(this, other)
    }

    operator fun times(other: Float): Tensor {
        return MulOp.apply(this, Tensor(data = floatArrayOf(other), shape = intArrayOf()))
    }

    operator fun div(other: Tensor): Tensor {
        return DivOp.apply(this, other)
    }

    operator fun div(other: Float): Tensor {
        return DivOp.apply(this, Tensor(data = floatArrayOf(other), shape = intArrayOf()))
    }

    operator fun minus(other: Tensor): Tensor {
        return SubtractOp.apply(this, other)
    }

    operator fun minus(other: Float): Tensor {
        return SubtractOp.apply(this, Tensor(data = floatArrayOf(other), shape = intArrayOf()))
    }

    operator fun unaryMinus(): Tensor {
        return NegOp.apply(this)
    }

    fun exp(): Tensor =
        ExpOp.apply(this)

    fun log(): Tensor =
        LogOp.apply(this)

    fun logSoftmax(axis: Int): Tensor =
        LogSoftmaxOp.apply(this, axis)

    fun Tensor.relu(): Tensor = ReLUOp.apply(this)

    fun Tensor.sigmoid(): Tensor = SigmoidOp.apply(this)

    fun mask(
        mask: Tensor,
        value: Float
    ): Tensor =
        MaskOp.apply(this, mask, value)

    fun backward() {

        require(numel == 1) {
            "backward() can only be called " +
                    "on a scalar tensor."
        }

        require(requiresGrad) {
            "Cannot call backward() on a tensor " +
                    "that does not require gradients."
        }

        val gradient =
            Tensor(
                data = floatArrayOf(1f),
                shape = intArrayOf()
            )

        backward(gradient)
    }

    override fun toString(): String {

        if (shape.isEmpty()) {
            return iterator().next().toString()
        }

        val iterator = iterator()
        val builder = StringBuilder()

        appendTensor(
            builder = builder,
            iterator = iterator,
            dimension = 0
        )

        return builder.toString()
    }

    private fun computeReshapeStrides(
        newShape: IntArray
    ): IntArray? {

        if (shape.isEmpty()) {
            return if (newShape.isEmpty()) {
                intArrayOf()
            } else {
                null
            }
        }

        val chunks = mutableListOf<Pair<Int, Int>>()

        var chunkSize = shape.last()
        var chunkStride = strides.last()

        for (i in shape.lastIndex - 1 downTo 0) {

            if (strides[i] == strides[i + 1] * shape[i + 1]) {
                chunkSize *= shape[i]
            } else {
                chunks.add(chunkSize to chunkStride)

                chunkSize = shape[i]
                chunkStride = strides[i]
            }
        }

        chunks.add(chunkSize to chunkStride)
        chunks.reverse()

        val newStrides = IntArray(newShape.size)

        var chunkIndex = 0
        var remainingInChunk =
            if (chunks.isNotEmpty()) chunks[0].first else 1

        chunkStride =
            if (chunks.isNotEmpty()) chunks[0].second else 1

        for (i in newShape.indices) {

            val dim = newShape[i]

            if (dim == 1) {
                newStrides[i] = remainingInChunk * chunkStride
                continue
            }

            if (remainingInChunk % dim != 0) {
                return null
            }

            remainingInChunk /= dim

            newStrides[i] = remainingInChunk * chunkStride

            if (remainingInChunk == 1) {

                chunkIndex++

                if (chunkIndex < chunks.size) {
                    remainingInChunk = chunks[chunkIndex].first
                    chunkStride = chunks[chunkIndex].second
                }
            }
        }

        return if (chunkIndex == chunks.size &&
            remainingInChunk == 1
        ) {
            newStrides
        } else {
            null
        }
    }

    private fun isContiguous(): Boolean {
        var expectedStride = 1

        for (i in shape.lastIndex downTo 0) {
            if (shape[i] == 0) {
                return true
            }

            if (strides[i] != expectedStride) {
                return false
            }

            expectedStride *= shape[i]
        }

        return true
    }

    private fun buildTopoSort(root: Tensor): List<Tensor> {
        val visited = mutableListOf<Tensor>()
        val order = mutableListOf<Tensor>()

        fun visit(node: Tensor) {
            if (!visited.add(node)) {
                return
            }

            val gradFn = node.gradFn ?: return

            for (edge in gradFn.nextEdges) {
                visit(edge.tensor)
            }

            order.add(node)
        }

        visit(root)

        return order.asReversed()
    }

    internal fun backward(
        gradOutput: Tensor
    ) {
        val order = buildTopoSort(this)

        val gradients = mutableMapOf<Tensor, Tensor>()

        gradients[this] = gradOutput

        for (tensor in order) {
            val grad = gradients[tensor] ?: continue

            val gradFn = tensor.gradFn ?: continue

            val inputGradients = gradFn.backward(grad)

            for (i in gradFn.nextEdges.indices) {
                val input = gradFn.nextEdges[i].tensor

                val inputGradient = inputGradients[i] ?: continue

                gradients[input] = gradients[input]?.let {
                    it + inputGradient
                } ?: inputGradient
            }
        }

        for ((tensor, gradient) in gradients) {
            if (tensor.requiresGrad) {
                tensor.grad = gradient
            }
        }
    }


    private fun appendTensor(
        builder: StringBuilder,
        iterator: Iterator<Float>,
        dimension: Int
    ) {

        builder.append("[")

        val size = shape[dimension]

        for (i in 0 until size) {

            if (dimension == shape.lastIndex) {

                builder.append(iterator.next())

            } else {

                appendTensor(
                    builder = builder,
                    iterator = iterator,
                    dimension = dimension + 1
                )
            }

            if (i < size - 1) {

                if (dimension == shape.lastIndex) {
                    builder.append(" ")
                } else {
                    builder.append("\n")
                    builder.append(" ".repeat(dimension + 1))
                }
            }
        }

        builder.append("]")
    }
}