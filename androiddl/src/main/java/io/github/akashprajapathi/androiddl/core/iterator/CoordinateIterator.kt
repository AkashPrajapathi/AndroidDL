package io.github.akashprajapathi.androiddl.core.iterator

class CoordinateIterator(
    private val shape: IntArray
) : Iterator<IntArray> {

    private val coordinate = IntArray(shape.size)

    private var finished = shape.any{it==0}

    override fun hasNext(): Boolean {
        return !finished
    }

    override fun next(): IntArray {

        if (!hasNext()) {
            throw NoSuchElementException()
        }

        val result = coordinate.copyOf()

        advance()

        return result
    }

    private fun advance() {
        for (dim in shape.lastIndex downTo 0) {
            coordinate[dim]++

            if (coordinate[dim] < shape[dim]) {
                return
            }

            coordinate[dim] = 0
        }

        finished = true
    }

}