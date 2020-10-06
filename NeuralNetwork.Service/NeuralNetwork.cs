using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NeuralNetwork.Service.ActivationFunctions;

namespace NeuralNetwork.Service
{
    public class NeuralNetwork
    {
        public Layer[] Layers { get; }
        public double[] OutputSignals { get; set; }

        public NeuralNetwork(int inputNeurons, int[] hiddenNeurons, int outputNeurons, int inputsNeuronInputsCount, IActivationFunction activationFunction )
        {
            Layers = new Layer[hiddenNeurons.Length + 2];
            Layers[0] = CreateLayer(LayerType.Input, inputNeurons, inputsNeuronInputsCount, new PassInitial());
            
            for(var i = 1; i < Layers.Length - 1; i++)
            {
                Layers[i] = CreateLayer(LayerType.Hidden, hiddenNeurons[i - 1], Layers[i - 1].Neurons.Length, activationFunction);
            }

            Layers[Layers.Length - 1] = CreateLayer(LayerType.Output, outputNeurons, Layers[Layers.Length - 2].Neurons.Length, activationFunction);
            RandomizeWeights(Layers);
        }

        public void Run(double[] inputSignals)
        {
            OutputSignals = Layers[0].FeedForward(inputSignals);
            
            for(var i = 1; i < Layers.Length; i++)
            {
                OutputSignals = Layers[i].FeedForward(OutputSignals);
            }
        }

        private Layer CreateLayer(LayerType layerType, int neuronsCount, int inputsPerNeuron, IActivationFunction activationFunction)
        {
            return new Layer(layerType, activationFunction, neuronsCount, inputsPerNeuron);
        }

        private void RandomizeWeights(Layer[] layers)
        {
            var random = new Random();

            for(var i = 1; i < layers.Length; i++)
            {
                for(var y = 0; y < layers[i].Neurons.Length; y++)
                {
                    var weights = new double[layers[i - 1].Neurons.Length];
                    weights.All(w => { w = random.NextDouble(); return true; });
                    layers[i].Neurons[y].SetWeights(weights);
                }
            }
        }
    }
}
