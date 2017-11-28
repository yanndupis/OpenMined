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

		public int addTensor(FloatTensor tensor) {
			tensors.Add (tensor.Id, tensor);
			return (tensors.Count);
		}

        public string processMessage(string json_message)
        {
            //Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

            Command msgObj = JsonUtility.FromJson<Command>(json_message);

            if (msgObj.functionCall == "create")
            {
                FloatTensor tensor = new FloatTensor(msgObj.data, msgObj.shape);
                tensor.Shader = shader;
                tensors.Add(tensor.Id, tensor);

                Debug.LogFormat("<color=magenta>createTensor:</color> {0}", string.Join(", ", tensor.Data));

	            string id = tensor.Id.ToString();

                return id;
            } else if (msgObj.functionCall == "deleteTensor")
            {
	            var tensor = tensors[msgObj.objectIndex];
	            tensors.Remove(msgObj.objectIndex);
	            tensor.Dispose();
            }
            else if (msgObj.objectType == "tensor")
            {
                if (msgObj.objectIndex > tensors.Count)
                {
                    return "Invalid objectIndex: " + msgObj.objectIndex;
                }
                else
                {
					FloatTensor tensor = this.getTensor(msgObj.objectIndex);    
                    // Process message's function
					return tensor.ProcessMessage(msgObj,this);
                }
            }
         
	        // If not executing createTensor or tensor function, return default error.
            return "SyftController.processMessage: Command not found.";
            
        }

        
    }
}