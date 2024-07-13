#ifndef SNN_MODEL_H
#define SNN_MODEL_H

#include <vector>
#include <random>

// Class representing a neuron in the spiking neural network
class Neuron {
public:
    Neuron();
    void stimulate(double input);  // Method to stimulate the neuron with an input
    void update_state();           // Method to update the state of the neuron
    bool has_spiked() const;       // Method to check if the neuron has spiked
    double get_potential() const;  // Method to get the current potential of the neuron

private:
    double potential;              // The potential of the neuron
    bool spiked;                   // Whether the neuron has spiked
    const double threshold = 1.0;  // The threshold potential for spiking
    const double decay = 0.9;      // The decay factor for the potential
};

// Class representing a synapse in the spiking neural network
class Synapse {
public:
    Synapse(Neuron* pre, Neuron* post, double weight);
    void transmit();              // Method to transmit signals from the pre-neuron to the post-neuron
    double get_weight() const;    // Method to get the weight of the synapse
    void update_weight(double delta); // Method to update the weight of the synapse

private:
    Neuron* pre_neuron;           // Pointer to the pre-synaptic neuron
    Neuron* post_neuron;          // Pointer to the post-synaptic neuron
    double weight;                // The weight of the synapse
};

// Class representing the spiking neural network model
class SNNModel {
public:
    SNNModel(int num_neurons);
    void stimulate(const std::vector<double>& inputs);  // Method to stimulate the network with inputs
    void update();                                      // Method to update the network state
    std::vector<double> get_output() const;             // Method to get the output potentials of the neurons
    void train(const std::vector<std::vector<double>>& data, int epochs);  // Method to train the network

private:
    std::vector<Neuron> neurons;                        // Vector of neurons in the network
    std::vector<Synapse> synapses;                      // Vector of synapses in the network
    std::mt19937 rng;                                   // Random number generator for initializing synapses
    void initialize_synapses();                         // Method to initialize the synapses
    void apply_stdp();                                  // Method to apply Spike-Timing-Dependent Plasticity (STDP)
};

#endif // SNN_MODEL_H
