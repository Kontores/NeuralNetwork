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
        public double[] OutputSignals { get; private set; }
        public double LearningRate { get; set; }

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

        private double[] Backpropagation(double[] expected, double[] inputSignals)
        {
            Run(inputSignals);

            var error = new double[OutputSignals.Length];
            for(var i = 0; i < error.Length; i++)
            {
                error[i] = OutputSignals[i] - expected[i];
                Layers.Last().Neurons[i].Learn(error[i], LearningRate);
            }
            
            for(var y = Layers.Length - 2; y > 0; y--)
            {
                var layer = Layers[y];
                var forwardLayer = Layers[y + 1];

                for(var i = 0; i < layer.Neurons.Length; i++)
                {
                    var neuron = layer.Neurons[i];                  
                    for(var k = 0; k < forwardLayer.Neurons.Length; k++)
                    {
                        var forwardNeuron = forwardLayer.Neurons[k];
                        var neuronError = forwardNeuron.Weights[i] * forwardNeuron.Delta;
                        neuron.Learn(neuronError, LearningRate);
                    }                    
                }
            }


            // returning square of error
            var result = new double[error.Length];
            result.All(r => { r *= r; return true; });
            return result;
        }

        private Layer CreateLayer(LayerType layerType, int neuronsCount, int inputsPerNeuron, IActivationFunction activationFunction)
        {
            return new Layer(layerType, activationFunction, neuronsCount, inputsPerNeuron);
        }

        private void RandomizeWeights(Layer[] layers)
        {
            // setting 1 weight value for input layer neurons
            var inputWeights = new double[Layers[1].Neurons.Length];
            inputWeights.All(w => { w = 1; return true; });
            Layers.First().Neurons.All(n => { n.SetWeights(inputWeights); return true; });

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
