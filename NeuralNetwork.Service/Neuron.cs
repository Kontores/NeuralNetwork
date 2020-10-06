using System;
using NeuralNetwork.Service.ActivationFunctions;

namespace NeuralNetwork.Service
{
    public class Neuron
    {
        private readonly IActivationFunction _activationFunction;
        public double[] Weights { get; private set; }
        public double[] InputSignals { get; private set; }
        public double OutputSignal { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int inputsCount, IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
            Weights = new double[inputsCount];
            InputSignals = new double[inputsCount];
        }

        public double Activate(double[] inputSignals)
        {
            if(inputSignals.Length != Weights.Length)
            {
                throw new Exception("Neuron weights number must be the same as input signals number passed to it");
            }

            OutputSignal = default(double);

            for(var i = 0; i < inputSignals.Length; i++)
            {
                InputSignals[i] = inputSignals[i];
                OutputSignal += inputSignals[i] * Weights[i];
            }

            return _activationFunction.Activate(OutputSignal);
        }

        public void SetWeights(double[] weights)
        {
            for(var i = 0; i < Weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public void Learn(double error, double learningRate)
        {
            Delta = error * _activationFunction.Derivative(OutputSignal);

            for(var i = 0; i < Weights.Length; i++)
            {
                var newWeight = Weights[i] - InputSignals[i] * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }
    }
}
