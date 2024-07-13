# Spiking Neural Network with Pattern Recognition

This project implements a Spiking Neural Network (SNN) in C++, integrates it with Python using `pybind11`, and provides a Python interface for dynamic pattern recognition and real-time data processing.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Details](#details)
  - [SNN Model](#snn-model)
  - [Python Interface](#python-interface)
- [Example](#example)
___
![img]()
___
## Installation

### Prerequisites

- C++ compiler
- Python 3
- `pybind11`
- `numpy`
- `scikit-learn`

### Build the C++ Library

1. Clone the repository:

   ```sh
   git clone https://github.com/HermiTech-LLC/2create.git
   cd 2create
   ```

2. Navigate into it build directory:

   ```sh
   cd build
   ```

3. Configure and build the project using CMake:

   ```sh
   cmake ..
   make
   ```

This will compile the C++ code and create a shared library that can be used in Python.

## Usage

### Python Interface

1. Ensure that the compiled shared library is in your Python path.
2. Import the necessary modules and use the `CognitiveModel` class as shown in the example below.

### Example

```python
import snn
import numpy as np
from Python.pattern_recon import CognitiveModel

num_neurons = 10
cognitive_model = CognitiveModel(num_neurons)

# Initial stimuli
initial_stimuli = np.random.rand(num_neurons).tolist()
print("Initial Stimuli:", initial_stimuli)

output = cognitive_model.stimulate(initial_stimuli)
print("Output after initial stimuli:", output)

# New data
new_data = np.random.rand(num_neurons).tolist()
print("New Data:", new_data)

updated_output = cognitive_model.stimulate(new_data)
print("Output after new data:", updated_output)

# Training example (placeholder data)
training_data = [np.random.rand(num_neurons).tolist() for _ in range(100)]
cognitive_model.train(training_data)

# Recognize pattern in new input
pattern = cognitive_model.recognize_pattern(new_data)
print("Recognized pattern:", pattern)

# Reconfigure with new insights
new_insights = [np.random.rand(num_neurons).tolist() for _ in range(50)]
cognitive_model.reconfigure(new_insights)
```

## Details

### SNN Model

The SNN model is implemented in C++ with three main classes:

- `Neuron`: Represents a neuron in the network.
- `Synapse`: Represents a synapse between two neurons.
- `SNNModel`: Manages a network of neurons and synapses, and handles stimulation, updating, and training.

The model includes basic spiking behavior and a simple form of Spike-Timing-Dependent Plasticity (STDP).

### Python Interface

The Python interface uses `pybind11` to expose the C++ classes and methods to Python. The `CognitiveModel` class in `pattern_recognition.py` integrates the SNN model with machine learning for pattern recognition and reconfiguration.

## Example

An example of using the `CognitiveModel` class is provided in the [Usage](#usage) section above. This example demonstrates how to stimulate the network, train it with data, recognize patterns, and reconfigure the network with new insights.
