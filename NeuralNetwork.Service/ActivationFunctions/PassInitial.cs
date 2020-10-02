using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Service.ActivationFunctions
{
    public class PassInitial : IActivationFunction
    {
        public double Activate(double inputSignal)
        {
            return inputSignal;
        }

        public double Derivative(double inputSignal)
        {
            return inputSignal;
        }
    }
}
