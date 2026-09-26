package io.github.akashprajapathi.androiddl.core.iterator

class TensorIterator(
    private val data: FloatArray,
    private val shape: IntArray,
    private val strides: IntArray,
    private val offset: Int
) : Iterator<Float> {

    private val coordinate = IntArray(shape.size)
    private var finished = false

    override fun hasNext(): Boolean {
        return !finished
    }

    override fun next(): Float {
        val physicalIndex = offset +
                coordinate.indices.sumOf {
                    coordinate[it] * strides[it]
                }

        val value = data[physicalIndex]

        advance()

        return value
    }

    private fun advance() {
        for (dim in coordinate.lastIndex downTo 0) {
            coordinate[dim]++

            if (coordinate[dim] < shape[dim]) {
                return
            }

            coordinate[dim] = 0
        }

        finished = true
    }
}