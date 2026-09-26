package io.github.akashprajapathi.androiddl.core.autograd

import io.github.akashprajapathi.androiddl.core.Tensor

class ReLUBackward(
    val input: Tensor
): GradFn {

    override val nextEdges = listOf(Edge(input))

    override fun backward(gradOutput: Tensor): List<Tensor?> {
        val inputGrad = FloatArray(input.numel)
        val inData = input.data
        val gradData = gradOutput.data

        for(i in inData.indices){
            inputGrad[i] = if(inData[i] > 0f) gradData[i] else 0f
        }

        val gradTensor = Tensor(inputGrad, input.shape.copyOf())

        return listOf(gradTensor)
    }
}