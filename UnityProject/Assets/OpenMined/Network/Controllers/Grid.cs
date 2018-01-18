using UnityEngine;
using System;
using System.Collections.Generic;
using System.Collections;
using OpenMined.Network.Utils;
using OpenMined.Network.Servers;
using OpenMined.Syft.Layer;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using OpenMined.UI;

namespace OpenMined.Network.Controllers
{
    public class Grid
    {

        private SyftController controller;

        public Grid(SyftController controller)
        {
            this.controller = controller;
        }

        public void Run(int inputId, int targetId, List<GridConfiguration> configurations, MonoBehaviour owner)
        {
            Debug.Log("Grid.Run");

            string ipfsHash = "";

            var inputTensor = controller.floatTensorFactory.Get(inputId);
            var targetTensor = controller.floatTensorFactory.Get(targetId);

            // write the input and target tensors to Ipfs
            var inputJob = new Ipfs();
            var targetJob = new Ipfs();

            var inputIpfsResponse = inputJob.Write(inputTensor.GetConfig());
            var targetIpfsResponse = targetJob.Write(targetTensor.GetConfig());

            Debug.Log("Input Hash: " + inputIpfsResponse.Hash);
            Debug.Log("Target Hash: " + targetIpfsResponse.Hash);

            var jobs = new string[configurations.Count];

            for (var i = 0; i < configurations.Count; ++i)
            {
                var config = configurations[i];
                var model = controller.getModel(config.model) as Sequential;
                var serializedModel = model.GetConfig();

                var configJob = new Ipfs();
                var ipfsJobConfig = new IpfsJobConfig(config.lr);
                var response = configJob.Write(new IpfsJob(serializedModel, ipfsJobConfig));

                jobs[i] = response.Hash;
            }

            var experiment = new IpfsExperiment(inputIpfsResponse.Hash, targetIpfsResponse.Hash, jobs);
            var experimentWriteJob = new Ipfs();
            var experimentResult = experimentWriteJob.Write(experiment);

            var request = new Request();
            owner.StartCoroutine(request.AddModel(owner, experimentResult.Hash));

            PollNext(owner, request);
        }

        void PollNext(MonoBehaviour owner, Request request)
        {
            owner.StartCoroutine(PollForGrads(owner, request));
        }

        IEnumerator PollForGrads(MonoBehaviour owner, Request request)
        {
            if (request.numModels > 0)
            {
                yield return request.GetNumModelGrads(owner, request.numModels);
            }
            else
            {
                yield return request.GetNumModels(owner);
            }
            
            yield return new WaitForSeconds(20);
            PollNext(owner, request);
        }

        public void TrainModel(MonoBehaviour owner, string input, string target, IpfsJob job, int modelId)
        {
            var tmpInput = Ipfs.Get<JToken>(input);
            var tmpTarget = Ipfs.Get<JToken>(target);

            var seq = CreateSequential(job.Model);

            var inputData = tmpInput.SelectToken("data").ToObject<float[]>();
            var inputShape = tmpInput.SelectToken("shape").ToObject<int[]>();
            var inputTensor = controller.floatTensorFactory.Create(_data: inputData, _shape: inputShape, _autograd: true);

            var targetData = tmpTarget.SelectToken("data").ToObject<float[]>();
            var targetShape = tmpTarget.SelectToken("shape").ToObject<int[]>();
            var targetTensor = controller.floatTensorFactory.Create(_data: targetData, _shape: targetShape, _autograd: true);

            var grad = controller.floatTensorFactory.Create(_data: new float[] { 1, 1, 1, 1 }, 
                                                            _shape: new int[] { 4, 1 });

            // 10 epochs .. make configurable
            for (var i = 0; i < 10; ++i) {
                var pred = seq.Forward(inputTensor);

                var loss = pred.Sub(targetTensor).Pow(2);
                loss.Backward(grad);

                foreach (var p in seq.getParameters())
                {
                    var pTensor = controller.floatTensorFactory.Get(p);
                    pTensor.Sub(pTensor.Grad, inline: true);
                }
            }

            var resultJob = new Ipfs();
            var config = new IpfsJobConfig(job.config.lr);
            var response = resultJob.Write(new IpfsJob(seq.GetConfig(), config));

            var req = new Request();
            owner.StartCoroutine(req.AddWeights(owner, modelId, response.Hash));
        }

        private Sequential CreateSequential(JToken model)
        {  
            var seq = new Sequential(controller);

            var layers = model.SelectToken("config").Children();
            foreach(var layer in layers)
            {
                var layerType = layer.SelectToken("class_name");
                switch (layerType.Value<String>())
                {
                    case "Linear":
                        // weight float tensor
                        var weightData = layer.SelectToken("config.weights.data").ToObject<float[]>();
                        var weightShape = layer.SelectToken("config.weights.shape").ToObject<int[]>();
                        var weightTensor = controller.floatTensorFactory.Create(_data: weightData, _shape: weightShape, _autograd: true);

                        // bias float tensor
                        var biasData = layer.SelectToken("config.bias.data").ToObject<float[]>();
                        var biasShape = layer.SelectToken("config.bias.shape").ToObject<int[]>();
                        var biasTensor = controller.floatTensorFactory.Create(_data: biasData, _shape: biasShape, _autograd: true);

                        var input = layer.SelectToken("config.input").ToObject<int>();
                        var output = layer.SelectToken("config.output").ToObject<int>();

                        var linear = new Linear(controller, input: input, output: output, weights: weightTensor, bias: biasTensor);
                        seq.AddLayer(linear);
                        break;
                    case "ReLU":
                        seq.AddLayer(new ReLU(controller));
                        break;
                    case "Log":
                        seq.AddLayer(new OpenMined.Syft.Layer.Log(controller));
                        break;
                    case "Dropout":
                        var rate = layer.SelectToken("config.rate").ToObject<float>();
                        var dropout = new Dropout(controller, rate);
                        seq.AddLayer(dropout);
                        break;
                    case "Softmax":
                        var dim = layer.SelectToken("config.dim").ToObject<int>();
                        seq.AddLayer(new Softmax(controller, dim));
                        break;
                }
            }

            return seq;
        }
    }

    public interface LayerDefinition {
        string GetLayerDefinition();
    }

    public class IpfsExperiment
    {
        public string input;
        public string target;
        public string[] jobs;

        public IpfsExperiment (string input, string target, string[] jobs)
        {
            this.input = input;
            this.target = target;
            this.jobs = jobs;
        }
    }

    public class IpfsJob
    {
        public JToken Model;
        public IpfsJobConfig config;

        public IpfsJob (JToken model, IpfsJobConfig config)
        {
            this.Model = model;
            this.config = config;
        }
    }

    public class IpfsJobConfig
    {
        [SerializeField] public float lr;

        public IpfsJobConfig(float lr)
        {
            this.lr = lr;
        }
    }
}
