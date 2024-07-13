#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "snn_model.h"

namespace py = pybind11;

PYBIND11_MODULE(snn, m) {
    py::class_<Neuron>(m, "Neuron")
        .def(py::init<>())
        .def("stimulate", &Neuron::stimulate)
        .def("update_state", &Neuron::update_state)
        .def("has_spiked", &Neuron::has_spiked)
        .def("get_potential", &Neuron::get_potential);

    py::class_<Synapse>(m, "Synapse");

    py::class_<SNNModel>(m, "SNNModel")
        .def(py::init<int>())
        .def("stimulate", &SNNModel::stimulate)
        .def("update", &SNNModel::update)
        .def("get_output", &SNNModel::get_output)
        .def("train", &SNNModel::train);
}
