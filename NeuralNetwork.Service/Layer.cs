using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NeuralNetwork.Service.ActivationFunctions;

namespace NeuralNetwork.Service
{
    public class Layer
    {
        private readonly IActivationFunction _activationFunction;
        public LayerType LayerType { get; }
        public Neuron[] Neurons { get; }
        public Layer(LayerType layerType, IActivationFunction activationFunction, int neuronsCount, int inputsPerNeuron)
        {
            _activationFunction = activationFunction;
            LayerType = layerType;
            Neurons = new Neuron[neuronsCount];
            Neurons.All(n => { n = new Neuron(inputsPerNeuron); return true; });
        }

        public double[] FeedForward(double[] inputSignals)
        {
            var outputSignals = new double[Neurons.Length];
            for(var i = 0; i < Neurons.Length; i++)
            {
                outputSignals[i] = Neurons[i].Activate(_activationFunction, inputSignals);
            }

            return outputSignals;
        }
    }
}
