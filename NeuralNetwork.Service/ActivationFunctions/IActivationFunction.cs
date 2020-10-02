using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Service.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Activate(double inputSignal);
        double Derivative(double inputSignal);
    }
}
