package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

class SigmoidBackward(
    val input: Tensor,
    val output: Tensor
) : GradFn {

    override val nextEdges = listOf(Edge(input))

    override fun backward(gradOutput: Tensor): List<Tensor?> {
        val ones = Tensor(FloatArray(output.numel) { 1f }, output.shape.copyOf())

        // dL/dx = gradOutput * output * (1 - output)
        val gradTensor = gradOutput * output * (ones - output)

        return listOf(gradTensor)
    }
}