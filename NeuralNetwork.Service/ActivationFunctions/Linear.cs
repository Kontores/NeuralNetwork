using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Service.ActivationFunctions
{
    public class Linear : IActivationFunction
    {
        public double Activate(double inputSignal)
        {
            return inputSignal < 0.5 ? 0 : 1;
        }
    }
}
