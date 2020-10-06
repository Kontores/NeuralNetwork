using System;
using NeuralNetwork.Service.ActivationFunctions;

namespace NeuralNetwork.Service
{
    public class Neuron
    {
        public double[] Weights { get; private set; }
        public double[] InputSignals { get; private set; }

        public Neuron(int inputsCount)
        {
            Weights = new double[inputsCount];
            InputSignals = new double[inputsCount];
        }

        public double Activate(IActivationFunction activationFunction, double[] inputSignals)
        {
            if(inputSignals.Length != Weights.Length)
            {
                throw new Exception("Neuron weights number must be the same as input signals number passed to it");
            }

            var outputSignal = default(double);

            for(var i = 0; i < inputSignals.Length; i++)
            {
                outputSignal += inputSignals[i] * Weights[i];
            }

            return activationFunction.Activate(outputSignal);
        }

        public void SetWeights(double[] weights)
        {
            for(var i = 0; i < Weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }
    }
}
