using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;

using OpenMined.Syft;
using OpenMined.Network.Utils;


namespace OpenMined.Network.Controllers {

	public class SyftController {

		[SerializeField]
		private ComputeShader shader;

		private List<FloatTensor> tensors;

		public SyftController(ComputeShader _shader)
		{
			shader = _shader;

			tensors = new List<FloatTensor>();

			test_inline_elementwise_subtract ();
		}
		
		private float[] randomWeights (int length) {
			Random.InitState (1);
			float[] syn0 = new float[length];
			for (int i = 0; i < length; i++) {
				syn0 [i] = 2 * Random.value - 1;
			}
			return syn0;
		}

		public string processMessage(string json_message) {

			//Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

			Command msgObj = JsonUtility.FromJson<Command>(json_message);

			if (msgObj.functionCall == "createTensor") {
				
				FloatTensor tensor = new FloatTensor (msgObj.data, msgObj.shape, shader);
				tensors.Add (tensor);

				Debug.LogFormat ("<color=magenta>createTensor:</color> {0}", string.Join (", ", tensor.data));

				tensor.gpu ();

				return msgObj.functionCall + ": OK";

			} else {

				if (msgObj.objectType == "tensor") {

					if (msgObj.objectIndex > tensors.Count) {
						return "Invalid objectIndex: " + msgObj.objectIndex;
					}

					FloatTensor tensor = tensors [msgObj.objectIndex];

					if (msgObj.functionCall == "init_sigmoid_matrix_multiply") {

						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];

						tensor.init_sigmoid_matrix_multiply (tensor_1);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "init_add_matrix_multiply") {

						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];

						tensor.init_add_matrix_multiply (tensor_1);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "init_weights") {

						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];

						tensor.init_weights (tensor_1);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "sigmoid_matrix_multiply") {

						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];
						FloatTensor tensor_2 = tensors [msgObj.tensorIndexParams[1]];

						tensor.sigmoid_matrix_multiply (tensor_1, tensor_2);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "reset_weights") {
						
						tensor.reset_weights ();

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "inline_elementwise_subtract") {
						
						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];

						tensor.inline_elementwise_subtract (tensor_1);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "multiply_derivative") {

						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];

						tensor.multiply_derivative (tensor_1);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "add_matrix_multiply") {

						FloatTensor tensor_1 = tensors [msgObj.tensorIndexParams[0]];
						FloatTensor tensor_2 = tensors [msgObj.tensorIndexParams[1]];

						tensor.add_matrix_multiply (tensor_1, tensor_2);

						return msgObj.functionCall + ": OK";

					} else if (msgObj.functionCall == "print") {

						tensor.cpu ();

						string data = string.Join (", ", tensor.data);
						Debug.LogFormat ("<color=cyan>print:</color> {0}", data);

						return data;
					}
				}

			}
			
			return "SyftController.processMessage: Command not found.";

		}

		void test_inline_elementwise_subtract () {

			float[] xdata = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
			int[] xshape = new int[] { 3, 3, 3, 3 };
			FloatTensor xtensor = new FloatTensor (xdata, xshape, shader);
			xtensor.gpu ();

			//float[] ydata = new float[] { 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1 };
			float[] ydata = new float[] { 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
			int[] yshape = new int[] { 3, 3, 3, 3 };
			FloatTensor ytensor = new FloatTensor (ydata, yshape, shader);
			ytensor.gpu ();

			xtensor.inline_elementwise_subtract (ytensor);

			xtensor.cpu ();
			ytensor.cpu ();

			Debug.LogFormat ("<color=green>SyftController.test_inline_elementwise_subtract: {0}</color>", string.Join (",", xtensor.data));
		}
	}
}
