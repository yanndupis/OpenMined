using System.Collections.Generic;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;


namespace OpenMined.Network.Controllers
{
    public class SyftController
    {
        [SerializeField] private ComputeShader shader;

        private Dictionary<int, FloatTensor> tensors;

        public SyftController(ComputeShader _shader)
        {
            shader = _shader;

            tensors = new Dictionary<int, FloatTensor>();
        }

        private float[] randomWeights(int length)
        {
            Random.InitState(1);
            float[] syn0 = new float[length];
            for (int i = 0; i < length; i++)
            {
                syn0[i] = 2 * Random.value - 1;
            }
            return syn0;
        }

        public string processMessage(string json_message)
        {
            //Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

            Command msgObj = JsonUtility.FromJson<Command>(json_message);

            if (msgObj.functionCall == "createTensor")
            {
                FloatTensor tensor = new FloatTensor(msgObj.data, msgObj.shape);
                tensor.Shader = shader;
                tensors.Add(tensor.Id, tensor);

                Debug.LogFormat("<color=magenta>createTensor:</color> {0}", string.Join(", ", tensor.Data));

	            string id = tensor.Id.ToString();

                return id;
            }
            else
            {
                if (msgObj.objectType == "tensor")
                {

	                //Below check needs additions/fix.
                    bool success = true;
                    if (msgObj.objectIndex > FloatTensor.CreatedObjectCount)

                    {
                        return "Invalid objectIndex: " + msgObj.objectIndex;
                    }

                    FloatTensor tensor = tensors[msgObj.objectIndex];

					if (msgObj.functionCall == "init_add_matrix_multiply") {
						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams [0]];
						tensor.ElementwiseMultiplication (tensor_1);
					}
                    else if (msgObj.functionCall == "inline_elementwise_subtract") {
						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams [0]];
						tensor.ElementwiseSubtract (tensor_1);
					}
                    else if (msgObj.functionCall == "multiply_derivative") {
						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams [0]];
						tensor.MultiplyDerivative (tensor_1);
					}
                    else if (msgObj.functionCall == "add_matrix_multiply") {
						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams [0]];
						FloatTensor tensor_2 = tensors [msgObj.tensorIndexParams [1]];
                        tensor.AddMatrixMultiply (tensor_1, tensor_2);
					}
                    else if (msgObj.functionCall == "print") {
						return tensor.Print();
					}
                    else if (msgObj.functionCall == "abs") {
						// calls the function on our tensor object
						tensor.Abs ();
					}
                    else if (msgObj.functionCall == "neg") {
						tensor.Neg ();
					}
                    else if (msgObj.functionCall == "add") {
						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams [0]];

						FloatTensor output = tensor_1.Add (tensor_1);
						tensors.Add(output.Id, output);
						string id = output.Id.ToString();
						return id;
					}

                    else if (msgObj.functionCall == "scalar_multiply")
                    {
                        //get the scalar, cast it and multiply
                        tensor.ScalarMultiplication((float)msgObj.tensorIndexParams[0]);

                    }
                    else
                    {
                        success = false;
                    }

                    if (success)
                    {
                        return msgObj.functionCall + ": OK";
                    }
                }
            }

            return "SyftController.processMessage: Command not found.";
        }
    }
}