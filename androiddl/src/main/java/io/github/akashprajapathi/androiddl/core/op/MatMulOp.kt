package io.github.akashprajapathi.androiddl.core.op

import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.core.autograd.MatMulBackward
import io.github.akashprajapathi.androiddl.core.broadcastShape
import io.github.akashprajapathi.androiddl.core.broadcastStrides
import io.github.akashprajapathi.androiddl.core.iterator.CoordinateIterator
import io.github.akashprajapathi.androiddl.core.outputIndex

object MatMulOp {

    fun apply(
        left: Tensor,
        right: Tensor
    ): Tensor {

        val leftRank = left.shape.size
        val rightRank = right.shape.size

        require(leftRank >= 2) {
            "MatMul requires left tensor to have rank >= 2."
        }

        require(rightRank >= 2) {
            "MatMul requires right tensor to have rank >= 2."
        }

        val m = left.shape[leftRank - 2]
        val k = left.shape[leftRank - 1]

        val rightK = right.shape[rightRank - 2]
        val n = right.shape[rightRank - 1]

        require(k == rightK) {
            "Cannot multiply ${left.shape.contentToString()} × " +
                    right.shape.contentToString()
        }

        val leftBatchShape =
            left.shape.copyOfRange(
                0,
                leftRank - 2
            )

        val rightBatchShape =
            right.shape.copyOfRange(
                0,
                rightRank - 2
            )

        val batchShape =
            broadcastShape(
                leftBatchShape,
                rightBatchShape
            )

        val leftBatchStrides =
            left.strides.copyOfRange(
                0,
                leftRank - 2
            )

        val rightBatchStrides =
            right.strides.copyOfRange(
                0,
                rightRank - 2
            )

        val effectiveLeftBatchStrides =
            broadcastStrides(
                inputShape = leftBatchShape,
                inputStrides = leftBatchStrides,
                outputShape = batchShape
            )

        val effectiveRightBatchStrides =
            broadcastStrides(
                inputShape = rightBatchShape,
                inputStrides = rightBatchStrides,
                outputShape = batchShape
            )

        val outputShape =
            batchShape + intArrayOf(m, n)

        val outputSize =
            outputShape.fold(1) { acc, dim ->
                acc * dim
            }

        val result =
            FloatArray(outputSize)

        val batchIterator =
            CoordinateIterator(batchShape)

        while (batchIterator.hasNext()) {

            val batchCoordinate =
                batchIterator.next()

            var leftBatchOffset =
                left.offset

            var rightBatchOffset =
                right.offset

            for (dimension in batchShape.indices) {

                leftBatchOffset +=
                    batchCoordinate[dimension] *
                            effectiveLeftBatchStrides[dimension]

                rightBatchOffset +=
                    batchCoordinate[dimension] *
                            effectiveRightBatchStrides[dimension]
            }

            val batchIndex =
                outputIndex(
                    batchCoordinate,
                    batchShape
                )

            val outputBatchOffset =
                batchIndex * m * n

            for (i in 0..<m) {

                for (j in 0..<n) {

                    var sum = 0f

                    for (contract in 0..<k) {

                        val leftIndex =
                            leftBatchOffset +
                                    i * left.strides[leftRank - 2] +
                                    contract *
                                    left.strides[leftRank - 1]

                        val rightIndex =
                            rightBatchOffset +
                                    contract *
                                    right.strides[rightRank - 2] +
                                    j *
                                    right.strides[rightRank - 1]

                        sum +=
                            left.data[leftIndex] *
                                    right.data[rightIndex]
                    }

                    val outputIndex =
                        outputBatchOffset +
                                i * n +
                                j

                    result[outputIndex] = sum
                }
            }
        }

        val output =
            Tensor(
                data = result,
                shape = outputShape,
                requiresGrad =
                    left.requiresGrad ||
                            right.requiresGrad
            )

        if (left.requiresGrad || right.requiresGrad) {
            output.gradFn =
                MatMulBackward(
                    left = left,
                    right = right
                )
        }

        return output
    }
}