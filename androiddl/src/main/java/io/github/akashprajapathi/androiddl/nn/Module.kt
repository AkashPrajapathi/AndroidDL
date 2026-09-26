package io.github.akashprajapathi.androiddl.nn

import io.github.akashprajapathi.androiddl.core.Tensor

abstract class Module {
    var isTraining: Boolean = true
        private set

    // Internal maps to maintain named parameters and submodules
    private val internalParameters = mutableMapOf<String, Parameter>()
    private val internalSubmodules = mutableMapOf<String, Module>()

    // Automatic naming counters for backward compatibility when names aren't provided
    private var paramCounter = 0
    private var submoduleCounter = 0

    // Callable interface for running forward pass
    abstract operator fun invoke(input: Tensor): Tensor

    // --- Parameter Registration ---

    // New named registration method
    protected fun registerParameter(name: String, param: Parameter): Parameter {
        internalParameters[name] = param
        return param
    }

    // Old positional registration method (preserves backward compatibility)
    protected fun registerParameter(param: Parameter): Parameter {
        val autoName = "param_${paramCounter++}"
        return registerParameter(autoName, param)
    }

    // --- Module Registration ---

    // New named registration method
    protected fun registerModule(name: String, module: Module): Module {
        internalSubmodules[name] = module
        return module
    }

    // Old positional registration method (preserves backward compatibility)
    protected fun registerModule(module: Module): Module {
        val autoName = "module_${submoduleCounter++}"
        return registerModule(autoName, module)
    }

    // --- Parameter Collection ---

    // Recursively collect all parameters from this module and all sub-modules
    fun parameters(): List<Tensor> {
        val params = mutableListOf<Tensor>()

        for ((_, p) in internalParameters) {
            params.add(p.value)
        }

        for ((_, m) in internalSubmodules) {
            params.addAll(m.parameters())
        }

        return params
    }

    // --- State Dict Serialization / Deserialization ---

    // Export state dictionary (parameter name -> float array data)
    fun stateDict(prefix: String = ""): Map<String, FloatArray> {
        val dict = mutableMapOf<String, FloatArray>()

        for ((name, param) in internalParameters) {
            val fullName = if (prefix.isEmpty()) name else "$prefix.$name"
            dict[fullName] = param.value.data.copyOf()
        }

        for ((name, subModule) in internalSubmodules) {
            val fullName = if (prefix.isEmpty()) name else "$prefix.$name"
            dict.putAll(subModule.stateDict(fullName))
        }

        return dict
    }

    // Load state dictionary values into existing parameters in-place
    fun loadStateDict(stateDict: Map<String, FloatArray>, prefix: String = "") {
        for ((name, param) in internalParameters) {
            val fullName = if (prefix.isEmpty()) name else "$prefix.$name"
            val savedData = stateDict[fullName]
                ?: throw IllegalArgumentException("Missing key in stateDict: $fullName")

            require(savedData.size == param.value.data.size) {
                "Size mismatch for $fullName: expected ${param.value.data.size}, got ${savedData.size}"
            }

            // In-place copy into parameter tensor's underlying array
            savedData.copyInto(param.value.data)
        }

        for ((name, subModule) in internalSubmodules) {
            val fullName = if (prefix.isEmpty()) name else "$prefix.$name"
            subModule.loadStateDict(stateDict, fullName)
        }
    }

    // --- Training & Evaluation Modes ---

    open fun train(mode: Boolean = true) {
        isTraining = mode
        for ((_, m) in internalSubmodules) {
            m.train(mode)
        }
    }

    fun eval() = train(false)

    open fun zeroGrad() {
        for (p in parameters()) {
            p.grad = null
        }
    }
}