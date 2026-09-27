# AndroidDL

**Deep Learning in Kotlin.** Built from tensors upward.

AndroidDL is an experimental, lightweight, CPU-based deep learning library written from the ground up in Kotlin. It provides a transparent, modular architecture designed to explore, understand, and implement the fundamental building blocks of modern deep learning and reinforcement learning systems entirely without external Python runtimes.

## Architecture Overview

AndroidDL follows a clean, bottom-up design:

```text
Tensor ──► Operations ──► AutoGrad ──► Neural Networks ──► Optimizers ──► Reinforcement Learning
```

1. **Tensor** — Multidimensional, strided memory structures supporting views, reshaping, and broadcasting.

2. **Operations** — Computational building blocks featuring integrated forward computations and analytical backward derivatives.

3. **Automatic Differentiation** — Reverse-mode autograd engine utilizing dynamic computational graphs.

4. **Neural Networks** — PyTorch-inspired modules such as `Module`, `Parameter`, `Sequential`, and `Linear`.

5. **Optimizers** — In-place parameter optimization with support for SGD, momentum tracking, and model checkpoint serialization.

6. **Reinforcement Learning** — Policy-gradient components built explicitly for interactive simulation loops such as REINFORCE.

---

## Core Components & Code Examples

### 1. Tensors & Operations

Tensors decouple logical shape and multidimensional indexing from physical memory layout via strides, allowing manipulations such as reshaping and permuting without redundant data copying.

```kotlin
import io.github.akashprajapathi.androiddl.core.Tensor

// Create a 2 × 3 tensor
val tensor = Tensor(
    data = floatArrayOf(
        1f, 2f, 3f,
        4f, 5f, 6f
    ),
    shape = intArrayOf(2, 3)
)

// Perform element-wise tensor operations
// with broadcasting support
val a = Tensor(
    floatArrayOf(1f, 2f, 3f),
    intArrayOf(1, 3)
)

val b = Tensor(
    floatArrayOf(10f, 20f, 30f),
    intArrayOf(1, 3)
)

val c = a + b
```

The result is:

```text
[11 22 33]
```

Tensor operations use the tensor's shape and stride information to determine how logical elements map to physical storage.

---

### 2. Neural Network Modules & Sequential Composition

AndroidDL follows a modular architecture inspired by frameworks such as PyTorch.

Model parameters are registered hierarchically, while submodules are recursively tracked for parameter collection, optimization, and checkpoint serialization.

```kotlin
import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.nn.*

val model = Sequential(
    Linear(
        inFeatures = 4,
        outFeatures = 8
    ),
    ReLU(),
    Linear(
        inFeatures = 8,
        outFeatures = 1
    )
)
```

This produces a simple network:

```text
Input (4)
   │
   ▼
Linear(4 → 8)
   │
   ▼
ReLU
   │
   ▼
Linear(8 → 1)
   │
   ▼
Output (1)
```

---

### 3. Training Loop & Automatic Differentiation

Training workflows couple forward evaluation, loss computation, reverse-mode automatic differentiation, and optimizer updates.

```kotlin
import io.github.akashprajapathi.androiddl.core.Tensor
import io.github.akashprajapathi.androiddl.optim.SGD

val model = MyCustomModel()

val optimizer = SGD(
    parameters = model.parameters(),
    lr = 0.01f
)

val xTrain = Tensor(
    floatArrayOf(
        0.5f,
        0.1f,
        -0.2f,
        0.8f
    ),
    intArrayOf(1, 4)
)

val yTarget = Tensor(
    floatArrayOf(1.0f),
    intArrayOf(1, 1)
)

// Training step
optimizer.zeroGrad()

// Forward pass
val prediction = model(xTrain)

// Compute loss
// Example: squared-error loss
val loss = (prediction - yTarget).let {
    it * it
}

// Backpropagate gradients
loss.backward()

// Update parameters
optimizer.step()
```

The training flow is:

```text
Input
  │
  ▼
Model
  │
  ▼
Prediction
  │
  ▼
Loss
  │
  ▼
backward()
  │
  ▼
Gradients
  │
  ▼
optimizer.step()
  │
  ▼
Updated Parameters
```

---

## Reinforcement Learning

AndroidDL is also designed for reinforcement-learning experiments.

One of the target use cases is **REINFORCE**, a policy-gradient algorithm where the neural network produces a policy over possible actions.

A typical workflow looks like:

```text
Environment
     │
     ▼
Observation
     │
     ▼
Policy Network
     │
     ▼
   Action
     │
     ▼
   Reward
     │
     ▼
  Returns
     │
     ▼
Policy Loss
     │
     ▼
 backward()
     │
     ▼
Optimizer
```

This allows the same tensor, operation, autograd, neural-network, and optimizer infrastructure to be used for reinforcement-learning experiments.

---

## Checkpointing

AndroidDL also provides checkpoint support for training state.

The intended checkpoint contents include:

```text
AndroidDL Checkpoint
├── Model Parameters
├── Optimizer State
├── Epoch
└── Losses
```

This allows training to be resumed without rebuilding the model state from scratch.

---

## Roadmap

- **Tensor Engine** — Strided views, broadcasting, reshaping, permuting, and logical iteration ✓

- **AutoGrad** — Reverse-mode automatic differentiation graph ✓

- **Optimizers** — SGD, momentum, and state checkpoint serialization ✓

- **Neural Networks** — Linear layers, activations (`ReLU`, `Sigmoid`), and structured sequential blocks ✓

- **Reinforcement Learning** — REINFORCE policy-gradient experiments and mobile integration ✓

---

## Installation

AndroidDL is available through Maven Central.

### Gradle Kotlin DSL

```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("io.github.akashprajapathi:androiddl:0.1.0")
}
```

### Maven Coordinates

```text
Group ID:    io.github.akashprajapathi
Artifact ID: androiddl
Version:     0.1.0
```

---

## Design Philosophy

AndroidDL is built **from tensors upward**.

The project focuses on making the fundamental mechanisms of deep learning explicit:

```text
Tensor
  ↓
Memory Layout
  ↓
Operations
  ↓
Derivatives
  ↓
Automatic Differentiation
  ↓
Neural Networks
  ↓
Optimization
  ↓
Reinforcement Learning
```

Rather than treating deep learning as a black box, AndroidDL explores the machinery underneath the abstractions.

---

## Project Status

AndroidDL is an experimental project and is actively evolving.

The API may change between releases as the underlying tensor engine, operations, neural-network components, and training infrastructure continue to develop.

---

## License

AndroidDL is released under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the complete license text.