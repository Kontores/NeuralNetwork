using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NeuralNetwork.Service.ActivationFunctions;

namespace NeuralNetwork.Service
{
    public class Layer
    {
        public LayerType LayerType { get; }
        public Neuron[] Neurons { get; }

        public Layer(LayerType layerType, int neuronsCount)
        {
            LayerType = layerType;
            Neurons = new Neuron[neuronsCount];
            IActivationFunction activationFunction;
            if(LayerType == LayerType.Input)
            {
                activationFunction = new Linear();
            }
            else
            {
                activationFunction = new Sigmoid();
            }
            Neurons.All(n => { n = new Neuron(activationFunction); return true; });
        }
    }
}
