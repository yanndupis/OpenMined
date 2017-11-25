using System.Collections;
using System.Collections.Generic;
using System.Linq;
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

        public ComputeShader Shader
        {
            get { return shader; }
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

		public FloatTensor getTensor (int index) {
			return tensors [index];
		}

        public ComputeShader GetShader()
        {
            return shader;
        }

        public void RemoveTensor(int index)
        {
            var tensor = tensors[index];
            tensors.Remove(index);
            tensor.Dispose();
        }

        public int addTensor(FloatTensor tensor) {
			tensors.Add (tensor.Id, tensor);
			return (tensors.Count - 1);
		}

        public string processMessage(string json_message)
        {
            //Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

            Command msgObj = JsonUtility.FromJson<Command>(json_message);

            Debug.LogFormat("<color=magenta>Message Received:</color> {0}", json_message);
            Debug.LogFormat("<color=magenta>Switch:</color> {0}", msgObj.objectType);
            switch (msgObj.objectType)
            {
                case "tensor":
                {
                    Debug.Log("Tensor switch running");
                    Debug.LogFormat("<color=magenta>Index is {0}.</color>", msgObj.objectIndex);

                    if (msgObj.objectIndex == 0 && msgObj.functionCall=="create" )
                    {
                        FloatTensor tensor = new FloatTensor(msgObj.data, msgObj.shape);
                        tensor.Shader = Shader;
                        this.addTensor(tensor);
                        Debug.LogFormat("<color=magenta>createTensor:</color> {0}", string.Join(", ", tensor.Data));
                        return tensor.Id.ToString();
                    }
                    else if( msgObj.objectIndex > tensors.Count)
                    {
                        Debug.Log("Why???");
                        return "Invalid objectIndex: " + msgObj.objectIndex;
                    }
                    else
                    {
                        Debug.LogFormat("<color=magenta>Getting tensor {0}.</color>", msgObj.objectIndex);
                        FloatTensor tensor = this.getTensor(msgObj.objectIndex);
                        // Process message's function
                        return tensor.processMessage(msgObj, this);
                    }
                }
                default: break;                
            }
            Debug.Log("Missed the switch");

            // If not executing createTensor or tensor function, return default error.
            return "SyftController.processMessage: Command not found.";            
        }        
    }
}