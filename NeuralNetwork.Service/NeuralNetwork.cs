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

        public NeuralNetwork(int inputNeurons, int[] hiddenNeurons, int outputNeurons, IActivationFunction inputLayerFunction, IActivationFunction hiddenLayerFunction, IActivationFunction outputLayerFunction)
        {
            Layers = new Layer[hiddenNeurons.Length + 2];
            Layers[0] = CreateLayer(LayerType.Input, inputNeurons, inputLayerFunction);
            
            for(var i = 1; i < Layers.Length - 1; i++)
            {
                Layers[i] = CreateLayer(LayerType.Hidden, hiddenNeurons[i - 1], hiddenLayerFunction);
            }

            Layers[Layers.Length - 1] = CreateLayer(LayerType.Output, outputNeurons, outputLayerFunction);
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

        private Layer CreateLayer(LayerType layerType, int neuronsCount, IActivationFunction activationFunction)
        {
            return new Layer(layerType, activationFunction, neuronsCount);
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
                    layers[i].Neurons[y].Weights = weights;
                }
            }
        }
    }
}
