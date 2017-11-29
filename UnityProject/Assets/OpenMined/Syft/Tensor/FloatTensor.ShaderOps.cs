using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private ComputeShader shader;

		[SerializeField]
		private static int AbsKernel;
        [SerializeField]
		private static int AbsKernel_;
		[SerializeField]
		private static int AddScalarKernel_;
		[SerializeField]
		private static int AddElemKernel_;
		[SerializeField]
		private static int AddScalarKernel;
		[SerializeField]
		private static int AddElemKernel;
		[SerializeField]
		private static int AddMMKernel_;
		[SerializeField]
		private static int CeilKernel;
		[SerializeField]
	    private static int FloorKernel_;
		[SerializeField]
		private static int MulScalarKernel_;
		[SerializeField]
		private static int MulElemKernel_;
		[SerializeField]
		private static int MulScalarKernel;
		[SerializeField]
		private static int MulElemKernel;
		[SerializeField]
		private static int NegateKernel;
		[SerializeField]
		private static int PowKernel;
		[SerializeField]
		private static int PowKernel_;
		[SerializeField]
		private static int SigmoidKernel_;
		[SerializeField]
		private static int SubElemKernel;
		[SerializeField]
		private static int TanhKernel;
		[SerializeField]
		private static int ZeroKernel_;

		public void initShaderKernels() {
			if (shader != null) {
				// save shaders and kernels
				AbsKernel = shader.FindKernel ("Abs");
				AbsKernel_ = shader.FindKernel ("Abs_");
				AddScalarKernel_ = shader.FindKernel ("AddScalar_");
				AddElemKernel_ = shader.FindKernel ("AddElem_");
				AddScalarKernel = shader.FindKernel ("AddScalar");
				AddElemKernel = shader.FindKernel ("AddElem");
				AddMMKernel_ = shader.FindKernel ("AddMM_");
				CeilKernel = shader.FindKernel ("Ceil");
				FloorKernel_ = shader.FindKernel ("Floor_");
				MulScalarKernel_ = shader.FindKernel ("MulScalar_");
				MulElemKernel_ = shader.FindKernel ("MulElem_");
				MulScalarKernel = shader.FindKernel ("MulScalar");
				MulElemKernel = shader.FindKernel ("MulElem");
				NegateKernel = shader.FindKernel ("Negate");
				PowKernel = shader.FindKernel ("Pow");
				PowKernel_ = shader.FindKernel ("Pow_");
				SigmoidKernel_ = shader.FindKernel ("Sigmoid_");
				SubElemKernel = shader.FindKernel ("SubElem");
				TanhKernel = shader.FindKernel ("Tanh");
				ZeroKernel_ = shader.FindKernel ("Zero_");
			}

		}

		public FloatTensor AbsGPU(FloatTensor result) {
			Debug.LogFormat("<color=blue>FloatTensor.AbsGPU dataOnGpu: {0}</color>", dataOnGpu);
			if (dataOnGpu) {
				shader.SetBuffer (AbsKernel, "abs_data", dataBuffer);
				shader.SetBuffer (AbsKernel, "abs_result", result.dataBuffer);
				shader.Dispatch (AbsKernel, this.size, 1, 1);
			}
			return result;
		}

		public void AbsGPU_() {
			Debug.LogFormat("<color=blue>FloatTensor.AbsGPU_ dataOnGpu: {0}</color>", dataOnGpu);
			if (dataOnGpu) {
				shader.SetBuffer (AbsKernel_, "abs_data_", dataBuffer);
				shader.Dispatch (AbsKernel_, this.size, 1, 1);
			}
		}

		public void AddScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.AddScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(AddScalarKernel_, value, "add_scalar_scalar_");

				shader.SetBuffer(AddScalarKernel_, "add_scalar_data_", dataBuffer);
				shader.Dispatch(AddScalarKernel_, this.size, 1, 1);

				valBuffer.Release();
			}
		}

		public void AddElemGPU_(FloatTensor tensor)
		{
			Debug.LogFormat("<color=blue>FloatTensor.AddElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				if (this.id != tensor.id) {
					shader.SetBuffer (AddElemKernel_, "add_elem_data_a_", dataBuffer);
					shader.SetBuffer (AddElemKernel_, "add_elem_data_b_", tensor.dataBuffer);
					shader.Dispatch (AddElemKernel_, this.size, 1, 1);
				} else {
					Debug.LogFormat("addition with itself should be multiplication instead", dataOnGpu);
					this.MulScalarGPU_(2);
				}

			}
		}

		public FloatTensor AddScalarGPU(float value, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.AddScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(AddScalarKernel, value, "add_scalar_scalar");

				shader.SetBuffer(AddScalarKernel, "add_scalar_data", dataBuffer);
				shader.SetBuffer(AddScalarKernel, "add_scalar_result", result.dataBuffer);
				shader.Dispatch(AddScalarKernel, this.size, 1, 1);

				valBuffer.Release();
			}
			return result;
		}

		public FloatTensor AddElemGPU(FloatTensor tensor, FloatTensor result)
		{
			
			Debug.LogFormat("<color=blue>FloatTensor.AddElemGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				// if tensors are not actually referring to the same tensor
				if (this.id != tensor.id) {
					shader.SetBuffer (AddElemKernel, "add_elem_data_a", this.DataBuffer);
					shader.SetBuffer (AddElemKernel, "add_elem_data_b", tensor.DataBuffer);
					shader.SetBuffer (AddElemKernel, "add_elem_data_result", result.DataBuffer);
					shader.Dispatch (AddElemKernel, this.size, 1, 1);
				} else {
					Debug.LogFormat("addition with itself should be multiplication instead", dataOnGpu);
					return this.MulScalarGPU (2, result);
				}
					

			}
			return result;
		}

		public void AddMatrixMultiplyGPU(FloatTensor tensor_1, FloatTensor tensor_2)
		{
			//Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
			shader.SetBuffer(AddMMKernel_, "addmm_data_a", dataBuffer);
			shader.SetBuffer(AddMMKernel_, "addmm_data_b", tensor_1.DataBuffer); //d
			shader.SetBuffer(AddMMKernel_, "addmm_data_c", tensor_2.DataBuffer);
			shader.Dispatch(AddMMKernel_, size, 1, 1);
		}

		public void InitAddMatrixMultiplyGpu(FloatTensor tensor_1)
		{
			var dim = new Dimensions[]
			{
				new Dimensions(tensor_1.shape.Length, tensor_1.shape[0])
			};

			var dimBuffer = new ComputeBuffer(dim.Length, dim[0].Stride());
			dimBuffer.SetData(dim);
			shader.SetBuffer(AddMMKernel_, "addmm_dimensions", dimBuffer);
		}

		public FloatTensor CeilGPU()
		{
			Debug.LogFormat("<color=blue>FloatTensor.ceil dataOnGpu: {0}</color>", dataOnGpu);

			if (!dataOnGpu) return this;
			var result = new FloatTensor(shape, this.shader, dataOnGpu);
			shader.SetBuffer(CeilKernel, "ceil_data", dataBuffer);
			shader.SetBuffer(CeilKernel, "ceil_result", result.DataBuffer);
			shader.Dispatch(CeilKernel, 1, 1, 1);
			return result;
		}

        	public void FloorGPU_()
        	{
            		if (DataOnGpu)
            		{
                		shader.SetBuffer(FloorKernel_, "floor_data_", dataBuffer);
                		shader.Dispatch(FloorKernel_, 1, 1, 1);
            		}
        	}


		public void MulScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.MulScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(MulScalarKernel_, value, "mul_scalar_scalar_");

				shader.SetBuffer(MulScalarKernel_, "mul_scalar_data_", dataBuffer);
				shader.Dispatch(MulScalarKernel_, this.size, 1, 1);

				valBuffer.Release();
			}
		}

		public void MulElemGPU_(FloatTensor tensor)
		{
			Debug.LogFormat("<color=blue>FloatTensor.MulElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				if (tensor.id != this.id) {
					shader.SetBuffer (MulElemKernel_, "mul_elem_data_a_", dataBuffer);
					shader.SetBuffer (MulElemKernel_, "mul_elem_data_b_", tensor.dataBuffer);
					shader.Dispatch (MulElemKernel_, this.size, 1, 1);
				} else {
					PowGPU_ (2);
				}

			}
		}

		public FloatTensor MulScalarGPU(float value, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.MulScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(MulScalarKernel, value, "mul_scalar_scalar");

				shader.SetBuffer(MulScalarKernel, "mul_scalar_data", dataBuffer);
				shader.SetBuffer(MulScalarKernel, "mul_scalar_result", result.dataBuffer);
				shader.Dispatch(MulScalarKernel, this.size, 1, 1);

				valBuffer.Release();
			}
			return result;
		}

		public FloatTensor MulElemGPU(FloatTensor tensor, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.MulElemGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				if (tensor.id != this.id) {
					shader.SetBuffer (MulElemKernel, "mul_elem_data_a", dataBuffer);
					shader.SetBuffer (MulElemKernel, "mul_elem_data_b", tensor.dataBuffer);
					shader.SetBuffer (MulElemKernel, "mul_elem_data_result", result.dataBuffer);
					shader.Dispatch (MulElemKernel, this.size, 1, 1);
				} else {
					return this.PowGPU(2, result);
				}

			}
			return result;
		}


        public FloatTensor NegateGPU()
        {
            if (dataOnGpu)
            {
				var result = new FloatTensor(shape, this.shader, dataOnGpu);
				shader.SetBuffer(NegateKernel, "negate_data", dataBuffer);
				shader.SetBuffer(NegateKernel, "negate_result", result.dataBuffer);
				shader.Dispatch(NegateKernel, 1, 1, 1);
                return result;
            }
            return this;
        }

		public FloatTensor PowGPU(float value, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.PowGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(PowKernel, value, "pow_scalar_scalar");

				shader.SetBuffer(PowKernel, "pow_scalar_data", dataBuffer);
				shader.SetBuffer(PowKernel, "pow_scalar_result", result.dataBuffer);
				shader.Dispatch(PowKernel, this.size, 1, 1);

				valBuffer.Release();
			}
			return result;
		}

		public void PowGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.PowGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(PowKernel_, value, "pow_scalar_scalar_");

				shader.SetBuffer(PowKernel_, "pow_scalar_data_", dataBuffer);
				shader.Dispatch(PowKernel_, this.size, 1, 1);

				valBuffer.Release();
			}
		}

        public void SigmoidGPU_()
        {
            if (dataOnGpu)
            {
                shader.SetBuffer(SigmoidKernel_, "sigmoid_data_", dataBuffer);
                shader.Dispatch(SigmoidKernel_, this.size, 1, 1);
            }
        }


		public FloatTensor SubElemGPU(FloatTensor other)
		{
			//Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);

			if (size == other.Size)
			{
				if (dataOnGpu && other.DataOnGpu)
				{
					var result = new FloatTensor(shape, this.shader, dataOnGpu);
					// correspond tensor buffers with shader kernel buffers
					shader.SetBuffer(SubElemKernel, "sub_elem_data_a", dataBuffer);
					shader.SetBuffer(SubElemKernel, "sub_elem_data_b", other.DataBuffer);
					shader.SetBuffer(SubElemKernel, "sub_elem_result", result.DataBuffer);
					shader.Dispatch(SubElemKernel, size, 1, 1);

					return result;
				}
				Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
			}
			Debug.Log("Tensors do not have the same number of elements!");
			return this;
		}

		public FloatTensor TanhGPU ()
		{
			var result = new FloatTensor(shape, this.shader, dataOnGpu);
			shader.SetBuffer(TanhKernel, "tanh_data", dataBuffer);
			shader.SetBuffer(TanhKernel, "tanh_result", result.DataBuffer);
			shader.Dispatch(TanhKernel, this.size, 1, 1);
			return result;
		}

        public void ZeroGPU_()
        {
			shader.SetBuffer(ZeroKernel_, "zero_data_", dataBuffer);
			shader.Dispatch(ZeroKernel_, 1, 1, 1);
        }

        public struct Dimensions
        {
            public int rows, columns;

            public Dimensions(int _rows, int _columns)
            {
                rows = _rows;
                columns = _columns;
            }

            public int Stride()
            {
                return 2 * sizeof(int);
            }
        }

        private ComputeBuffer SendFloatToGpu(int kernel, float value, string name)
        {
            float[] scalarArray = new float[1];
            scalarArray[0] = value;

            var scalarBuffer = new ComputeBuffer(1, sizeof(float));
            scalarBuffer.SetData(scalarArray);
			shader.SetBuffer(kernel, name, scalarBuffer);

            return scalarBuffer;
        }

    }
}
