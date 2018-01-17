﻿using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using OpenMined.Network.Servers;
using Newtonsoft.Json.Linq;

namespace OpenMined.Syft.Layer
{
    
    public class Linear: Layer, LayerDefinition
	{
		private int _input;
		private int _output;

        [SerializeField] string name = "linear";
        [SerializeField] public readonly FloatTensor _weights;
        [SerializeField] FloatTensor _bias;
		
		public Linear (SyftController _controller, int input, int output, string initializer="Xavier")
		{
            init(this.name);

			this.controller = _controller;
			
			_input = input;
			_output = output;
			
			int[] weightShape = { input, output };
            var weights = initializer == "Xavier" ? controller.RandomWeights(input * output, input) : controller.RandomWeights(input * output);
			_weights = controller.floatTensorFactory.Create(_shape: weightShape, _data: weights, _autograd: true, _keepgrads: true);

			int[] biasShape = {1,output};
			_bias = controller.floatTensorFactory.Create(_shape:biasShape, _autograd: true);

			parameters.Add(_weights.Id);
			parameters.Add(_bias.Id);
			
			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);

		}

        public override FloatTensor Forward(FloatTensor input)
		{
			
			FloatTensor unbiased_output = input.MM(_weights);
			FloatTensor output = unbiased_output.Add(_bias.Expand(unbiased_output.Shape).Contiguous());
			
			activation = output.Id;
		
			return output;
		}

        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }

		public override int getParameterCount(){return _weights.Size + _bias.Size;}

	   	public override JToken GetConfig()
        {
		    var config = new JObject
			{
			    { "name", "linear" },
				{ "trainable", true },
				{ "dtype", "float32" }, 
				{ "output", _output },
                { "input", _input },
				{ "activation", "linear" },
				{ "use_bias", true },
				{
				    "kernel_initializer", new JObject
					{
					    { "class_name", "VarianceScaling" },
						{ 
						    "config", new JObject
						  	{
							    { "scale", 1.0 },
							  	{ "mode", "fan_avg" },
							  	{ "distribution", "uniform" },
							  	{ "seed", null }
						  	}
						}
					}
				},
				{ 
				    "bias_initializer", new JObject
					{
		          	    { "class_name", "Zeros"},
		          		{ "config", new JObject() }
		          	}
		        },
				{ "kernel_regularizer", null },
		        { "bias_regularizer", null },
		        { "activity_regularizer", null },
		        { "kernel_constraint", null },
		        { "bias_constraint", null }
				};

				return config;
		}

	}
}

