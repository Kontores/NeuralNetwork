using System;
using NeuralNetwork.Service.ActivationFunctions;

namespace NeuralNetwork.Service
{
    public class Neuron
    {
        private readonly IActivationFunction _activationFunction;
        public double[] Weights { get; set; }
        public Neuron(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }
        public double Activate(double[] inputs)
        {
            if(inputs.Length != Weights.Length)
            {
                throw new Exception("Neuron weights number must be the same as input signals number passed to it");
            }

            var totalSignal = default(double);

            for(var i = 0; i < inputs.Length; i++)
            {
                totalSignal += inputs[i] * Weights[i];
            }

            return _activationFunction.Activate(totalSignal);
        }
    }
}
