# Point Detector (A Simple C++ Neural Network Demo)

## Summary

A simple demo of artifical neural networks (simple perceptrons and multilayer perceptrons )in action, written in C++.
Based on the mathematics presented in this paper: http://home.thep.lu.se/pub/Preprints/91/lu_tp_91_23.pdf
Uses supervised learning via Backpropagation, and the transfer function used is tanh.

## Description

This demo does the following in order and outputs relevant statistics to screen:
- Teaches a simple perceptron AND logic, starting from random weights.
- Teaches a simple perceptron OR logic, starting from random weights.
- Teaches a simple multilayer perceptron NAND logic, starting from random weights.
- Teaches a 2 hidden layer multilayer perceptron (6 neurons in layer 1, 4 in layer 2) to detect the point {x1 = 0.7, x2 = 0.3} for a range of x1/x2 of 0 to 1.

Please note: As the initial weights are randomly generated in this case, learning does not always converge (or finish fast enough). If the network takes too many learning iterations, this demo will re-initisate the network weights to retry learning until a set number of initialisations before it gives up. Normally, a set of weights should manage to converge within 10 initialisations, but results will differ due to randomness at work.

## Building

Uses C++11 syntax, hence please remember to use the appropriate compiler flags for your compiler. (eg. g++ -std=c++11)