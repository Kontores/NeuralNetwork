using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Service.ActivationFunctions
{
    public class Sigmoid : IActivationFunction
    {
        public double Activate(double inputSignal)
        {
            return 1 / (1 + Math.Pow(Math.E, -inputSignal));
        }

        public double Derivative(double inputSignal)
        {
            var sigmoid = Activate(inputSignal);
            return sigmoid / (1 - sigmoid);
        }
    }
}
