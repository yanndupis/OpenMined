using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class FloatTensor {

	public float[] data;
	private int[] shape;
	private int _size;
	private int ndim;

	private bool data_on_gpu;

	public ComputeBuffer data_buffer;
	private ComputeBuffer shape_buffer;

	[SerializeField]
	private ComputeShader shader;

	private int ScalarMultMain;
	private int ElementwiseMultMain;
	private int ElementwiseSubtractMain;
	private int SigmoidMatrixMultiply;
	private int MultiplyDerivative;
	private int AddMatrixMultiply;
	private int ResetWeights;


	public FloatTensor(float[] _data, int[] _shape, ComputeShader _shader)
	{
		shape = new int[_shape.Length];
		data_on_gpu = false;

		// infer and store dimension data from data and shape
		// copy because otherwise two vectors might share the same
		// underlying information.
		_size = 0;
		ndim = 0;
		for(int i=0; i<_shape.Length; i++) {
			shape [i] = _shape [i];
			_size += _shape[i];
			ndim += 1;
		}

		// copy data into new object
		data = new float[_size];
		for (int i = 0; i < _size; i++) {
			data [i] = _data [i];
		}

		// save shaders and kernels
		shader = _shader;
		ScalarMultMain = shader.FindKernel ("ScalarMultMain");
		ElementwiseMultMain = shader.FindKernel ("ElementwiseMultMain");
		ElementwiseSubtractMain = shader.FindKernel ("ElementwiseSubtractMain");
		SigmoidMatrixMultiply = shader.FindKernel ("SigmoidMatrixMultiply");
		MultiplyDerivative = shader.FindKernel ("MultiplyDerivative");
		AddMatrixMultiply = shader.FindKernel ("AddMatrixMultiply");
		ResetWeights = shader.FindKernel ("ResetWeights");

	}

	public void inline_elementwise_mult(FloatTensor other) {
		Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_mult data_on_gpu: {0}</color>", data_on_gpu);

		if (size () == other.size ()) { 
			if (data_on_gpu && other.data_is_on_gpu ()) {

				// correspond tensor buffers with shader kernel buffers
				shader.SetBuffer (ElementwiseMultMain, "data_a", data_buffer);
				shader.SetBuffer (ElementwiseMultMain, "data_b", other.data_buffer);

				shader.Dispatch(ElementwiseMultMain, 1, 1, 1);

			} else if (!data_on_gpu && !other.data_is_on_gpu ()) {
				for (int i = 0; i < _size; i++) {
					data [i] = data [i] * other.data [i];
				}
			} else {
				Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
			}
		} else {
			Debug.Log("Tensors do not have the same number of elements!");
		}
	}

	public void scalar_mult(float value) {
		Debug.LogFormat("<color=blue>FloatTensor.scalar_mult data_on_gpu: {0}</color>", data_on_gpu);

		if (data_on_gpu) {

			ComputeBuffer scalar_buffer = send_float_to_gpu (value, "temp_scalar");

			shader.SetBuffer (ScalarMultMain, "data", data_buffer);
			shader.Dispatch(ScalarMultMain, 1, 1, 1);

			scalar_buffer.Release ();

		} else {
			for (int i = 0; i < _size; i++)
			{
				data [i] = data [i] * value;
			}
		}
	}

	public void inline_elementwise_subtract(FloatTensor other) {
		//Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract data_on_gpu: {0}</color>", data_on_gpu);

		if (size () == other.size ()) {
			if (data_on_gpu && other.data_is_on_gpu ()) {

				// correspond tensor buffers with shader kernel buffers
				shader.SetBuffer (ElementwiseSubtractMain, "data_c", data_buffer);
				shader.SetBuffer (ElementwiseSubtractMain, "data_d", other.data_buffer);

				shader.Dispatch(ElementwiseSubtractMain, _size, 1, 1);

			} else if (!data_on_gpu && !other.data_is_on_gpu ()) {
				for (int i = 0; i < _size; i++) {
					data [i] = data [i] - other.data [i];
				}
			} else {
				Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
			}
		} else {
			Debug.Log("Tensors do not have the same number of elements!");
		}
	}

	public void init_sigmoid_matrix_multiply(FloatTensor tensor_1) {

		Dimensions[] dim = new Dimensions[] {
			new Dimensions (tensor_1.shape.Length, tensor_1.shape [0])
		};

		ComputeBuffer dim_buffer = new ComputeBuffer (dim.Length, dim[0].stride());
		dim_buffer.SetData (dim);
		shader.SetBuffer (SigmoidMatrixMultiply, "dimensions_a", dim_buffer);
	}

	public void sigmoid_matrix_multiply(FloatTensor tensor_1, FloatTensor tensor_2) {
		//Debug.LogFormat("<color=blue>FloatTensor.sigmoid_matrix_multiply data_on_gpu: {0}</color>", data_on_gpu);
		shader.SetBuffer (SigmoidMatrixMultiply, "data_e", data_buffer);
		shader.SetBuffer (SigmoidMatrixMultiply, "data_f", tensor_1.data_buffer);
		shader.SetBuffer (SigmoidMatrixMultiply, "data_g", tensor_2.data_buffer);
		shader.Dispatch (SigmoidMatrixMultiply, _size, 1, 1);
	}

	public void multiply_derivative(FloatTensor tensor_1) {
		//Debug.LogFormat("<color=blue>FloatTensor.multiply_derivative data_on_gpu: {0}</color>", data_on_gpu);
		shader.SetBuffer (MultiplyDerivative, "data_h", data_buffer);
		shader.SetBuffer (MultiplyDerivative, "data_i", tensor_1.data_buffer);
		shader.Dispatch (MultiplyDerivative, _size, 1, 1);
	}

	public struct Dimensions {
		public int rows, columns;

		public Dimensions (int _rows, int _columns) {
			rows = _rows;
			columns = _columns;
		}

		public int stride() {
			return 2 * sizeof(int);
		}
	}

	public void init_add_matrix_multiply(FloatTensor tensor_1) {

		Dimensions[] dim = new Dimensions[] {
			new Dimensions (tensor_1.shape.Length, tensor_1.shape [0])
		};

		ComputeBuffer dim_buffer = new ComputeBuffer (dim.Length, dim[0].stride());
		dim_buffer.SetData (dim);
		shader.SetBuffer (AddMatrixMultiply, "dimensions_b", dim_buffer);
	}

	public void add_matrix_multiply(FloatTensor tensor_1, FloatTensor tensor_2) {
		//Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply data_on_gpu: {0}</color>", data_on_gpu);
		shader.SetBuffer (AddMatrixMultiply, "data_j", data_buffer);
		shader.SetBuffer (AddMatrixMultiply, "data_k", tensor_1.data_buffer); //d
		shader.SetBuffer (AddMatrixMultiply, "data_l", tensor_2.data_buffer);
		shader.Dispatch (AddMatrixMultiply, _size, 1, 1);
	}

	public void init_weights(FloatTensor save_tensor) {
		shader.SetBuffer (ResetWeights, "weights", data_buffer);
		shader.SetBuffer (ResetWeights, "original_weights", save_tensor.data_buffer);
	}

	public void reset_weights() {
		shader.Dispatch (ResetWeights, _size, 1, 1);
	}

	public bool data_is_on_gpu() {
		return data_on_gpu;
	}

	public int size() {
		return _size;
	}

	public void print() {

		if (data_on_gpu) {
			copy_gpu2cpu ();
		}

		for (int i = 0; i < _size; i++)
		{
			Debug.Log(data[i]);
		}

		if (data_on_gpu) {
			erase_cpu ();
		}

	}

	public void gpu () {

		if (!data_on_gpu) {

			copy_cpu2gpu ();
			erase_cpu ();

			data_on_gpu = true;
		}
	}

	public void cpu() {
		if (data_on_gpu) {

			copy_gpu2cpu ();
			erase_gpu();

			data_on_gpu = false;
		} 
	}

	private void copy_cpu2gpu() {
		data_buffer = new ComputeBuffer (_size, sizeof(float));
		shape_buffer = new ComputeBuffer (ndim, sizeof(int));

		data_buffer.SetData (data);	
		shape_buffer.SetData (shape);
	}

	private void erase_cpu() {
		data = null;
	}

	private void copy_gpu2cpu() {

		data = new float[_size];
		data_buffer.GetData(data);
	}

	private void erase_gpu() {
		data_buffer.Release ();
		shape_buffer.Release ();
	}

	private ComputeBuffer send_float_to_gpu(float value, string name) {
		float[] scalar_array = new float[1];
		scalar_array[0] = value;

		ComputeBuffer scalar_buffer = new ComputeBuffer (1, sizeof(float));
		scalar_buffer.SetData (scalar_array);	
		shader.SetBuffer (ScalarMultMain, name, scalar_buffer);

		return scalar_buffer;
	}

}

public class Command
{
	// given that SyftController keeps lists of objects of base types 
	// (at the time of writing this is only Tensors) then this command
	// selects one of these generic types and performs a command.
	public string objectType; // i.e. "tensor"
	public int objectIndex; //

	// name of the function to be called
	public string functionCall;

}

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

	public string processMessage(string json_message) {

		Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);
	
		Command cmd = JsonUtility.FromJson<Command>(json_message);

//		if (cmd.objectType == "FloatTensor") {
//			if (cmd.functionCall == "FloatTensor") {
//				FloatTensor x = new FloatTensor (fdata, fshape, shader);
//				
//				tensors.Add (x);
//			}
//		}
		Debug.Log("Object Type:" + (cmd.objectType));
		return "WE DID IT!!!";




//		var splittedStrings = message.Split(' ');
//
//		if (splittedStrings [0] == "0") { // command to create a new object of some type
//
//			Debug.Log("<color=green>SyftController.processMessage: Create a tensor object</color>");
//
//			float[] fdata = new float[splittedStrings.Length - 1];
//			for (int i = 0; i < splittedStrings.Length - 1; i++) {
//				fdata [i] = float.Parse (splittedStrings [i + 1]);
//			}
//
//			int[] fshape = new int[1];
//			fshape [0] = splittedStrings.Length - 1;
//
//			FloatTensor x = new FloatTensor (fdata, fshape, shader);
//
//			tensors.Add (x);
//
//			string created = string.Join(",", x.data);
//			Debug.LogFormat("<color=green>SyftController.processMessage: FloatTensor created: {0}</color>", created);
//
//		} else if (splittedStrings [0] == "1") { // command to do something with a Tensor object
//
//			Debug.Log("<color=green>SyftController.processMessage: Execute a tensor object command</color>");
//
//			int tensor_index = int.Parse (splittedStrings [1]);
//
//			FloatTensor tensor = tensors [tensor_index];
//
//			int message_offset = 2;
//
//			string command = splittedStrings [message_offset];
//			Debug.LogFormat("<color=green>SyftController.processMessage command: {0}</color>", command);
//
//			if (command == "0") { // command to call scalar_mult
//				float factor = (float)int.Parse (splittedStrings [message_offset + 1]);
//				Debug.LogFormat ("<color=green>SyftController.processMessage factor: {0}</color>", factor);
//
//				string before = string.Join (",", tensor.data);
//
//				tensor.scalar_mult (factor);
//
//				string after = string.Join (",", tensor.data);
//
//				Debug.LogFormat ("<color=green>SyftController.processMessage answer: {0} * {1} = {2}</color>", before, factor, after);
//
//			} else if (command == "1") { // command to call inline_elementwise_subtract
//				int other_tensor_index = int.Parse (splittedStrings [message_offset + 1]);
//				Debug.LogFormat ("<color=green>SyftController.processMessage other_tensor_index: {0}</color>", other_tensor_index);
//
//				FloatTensor other_tensor = tensors [other_tensor_index];
//
//				string before = string.Join (",", tensor.data);
//
//				string other_tensor_data = string.Join (",", other_tensor.data);
//
//				tensor.inline_elementwise_subtract (other_tensor);
//
//				string after = string.Join (",", tensor.data);
//
//				Debug.LogFormat ("<color=green>SyftController.processMessage answer: {0} - {1} = {2}</color>", before, other_tensor_data, after);
//
//			}
//		}

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
