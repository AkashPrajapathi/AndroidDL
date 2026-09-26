package io.github.akashprajapathi.androiddl.io

import io.github.akashprajapathi.androiddl.nn.Module
import io.github.akashprajapathi.androiddl.optim.Optimizer
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

object ModelIO {

    // Save Checkpoint: Model State + Optimizer State
    fun saveCheckpoint(module: Module, optimizer: Optimizer, filePath: String) {
        val modelDict = module.stateDict()
        val optimDict = optimizer.stateDict()

        File(filePath).outputStream().buffered().use { fileOut ->
            DataOutputStream(fileOut).use { out ->
                // Write Model Parameters
                out.writeInt(modelDict.size)
                for ((key, value) in modelDict) {
                    out.writeUTF(key)
                    out.writeInt(value.size)
                    for (f in value) out.writeFloat(f)
                }

                // Write Optimizer Parameters
                out.writeInt(optimDict.size)
                for ((key, value) in optimDict) {
                    out.writeUTF(key)
                    out.writeInt(value.size)
                    for (f in value) out.writeFloat(f)
                }
            }
        }
    }

    // Load Checkpoint: Model State + Optimizer State
    fun loadCheckpoint(module: Module, optimizer: Optimizer, filePath: String) {
        val modelDict = mutableMapOf<String, FloatArray>()
        val optimDict = mutableMapOf<String, FloatArray>()

        File(filePath).inputStream().buffered().use { fileIn ->
            DataInputStream(fileIn).use { input ->
                // Read Model Parameters
                val numModelParams = input.readInt()
                for (i in 0 until numModelParams) {
                    val key = input.readUTF()
                    val size = input.readInt()
                    val data = FloatArray(size) { input.readFloat() }
                    modelDict[key] = data
                }

                // Read Optimizer Parameters
                val numOptimParams = input.readInt()
                for (i in 0 until numOptimParams) {
                    val key = input.readUTF()
                    val size = input.readInt()
                    val data = FloatArray(size) { input.readFloat() }
                    optimDict[key] = data
                }
            }
        }

        module.loadStateDict(modelDict)
        optimizer.loadStateDict(optimDict)
    }
}