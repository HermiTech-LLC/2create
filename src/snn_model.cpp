#include "snn_model.h"
#include <algorithm>
#include <iostream>
#include <cmath>

// Neuron class implementation

Neuron::Neuron() : potential(0.0), spiked(false) {}

void Neuron::stimulate(double input) {
    potential += input;
}

void Neuron::update_state() {
    if (potential >= threshold) {
        spiked = true;
        potential = 0.0;
    } else {
        spiked = false;
        potential *= decay;
    }
}

bool Neuron::has_spiked() const {
    return spiked;
}

double Neuron::get_potential() const {
    return potential;
}

// Synapse class implementation

Synapse::Synapse(Neuron* pre, Neuron* post, double weight)
    : pre_neuron(pre), post_neuron(post), weight(weight) {}

void Synapse::transmit() {
    if (pre_neuron->has_spiked()) {
        post_neuron->stimulate(weight);
    }
}

double Synapse::get_weight() const {
    return weight;
}

void Synapse::update_weight(double delta) {
    weight += delta;
}

// SNNModel class implementation

SNNModel::SNNModel(int num_neurons) : neurons(num_neurons), rng(std::random_device{}()) {
    initialize_synapses();
}

void SNNModel::initialize_synapses() {
    std::uniform_real_distribution<double> weight_dist(0.1, 1.0);
    for (size_t i = 0; i < neurons.size(); ++i) {
        for (size_t j = 0; j < neurons.size(); ++j) {
            if (i != j) {
                synapses.emplace_back(&neurons[i], &neurons[j], weight_dist(rng));
            }
        }
    }
}

void SNNModel::stimulate(const std::vector<double>& inputs) {
    for (size_t i = 0; i < neurons.size() && i < inputs.size(); ++i) {
        neurons[i].stimulate(inputs[i]);
    }
}

void SNNModel::update() {
    for (auto& neuron : neurons) {
        neuron.update_state();
    }
    for (auto& synapse : synapses) {
        synapse.transmit();
    }
    apply_stdp();
}

std::vector<double> SNNModel::get_output() const {
    std::vector<double> output;
    for (const auto& neuron : neurons) {
        output.push_back(neuron.get_potential());
    }
    return output;
}

void SNNModel::train(const std::vector<std::vector<double>>& data, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            stimulate(sample);
            update();
        }
    }
}

void SNNModel::apply_stdp() {
    for (auto& synapse : synapses) {
        if (synapse.get_weight() > 0) {
            if (synapse.pre_neuron->has_spiked() && synapse.post_neuron->has_spiked()) {
                synapse.update_weight(0.01);
            } else if (synapse.pre_neuron->has_spiked() && !synapse.post_neuron->has_spiked()) {
                synapse.update_weight(-0.01);
            }
        }
    }
}
