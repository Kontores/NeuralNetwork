using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Service.ActivationFunctions
{
    public class Binary : IActivationFunction
    {
        public double Activate(double inputSignal)
        {
            return inputSignal < 0.5 ? 0 : 1;
        }

        public double Derivative(double inputSignal)
        {
            return Activate(inputSignal);
        }
    }
}
