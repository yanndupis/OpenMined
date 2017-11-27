using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private ComputeShader shader;


        [SerializeField]
		private static int AbsKernel_;
		private static int AddScalarKernel_;
		private static int AddElemKernel_;
		private static int AddMMKernel_;
		private static int CeilKernel;
	    	private static int FloorKernel_;
		private static int MultElemKernel;
		private static int MultScalarKernel_;
		private static int NegateKernel;
		private static int SubElemKernel;
		private static int ZeroKernel_;

        public ComputeShader Shader
        {
            get { return shader; }
            set
            {
                shader = value;

                // save shaders and kernels
				AbsKernel_ = shader.FindKernel("Abs_");
				AddScalarKernel_ = shader.FindKernel("AddScalar_");
				AddElemKernel_ = shader.FindKernel("AddElem_");
				AddMMKernel_ = shader.FindKernel("AddMM_");
				CeilKernel = shader.FindKernel("Ceil");
                		FloorKernel_ = shader.FindKernel("Floor_");
				MultElemKernel = shader.FindKernel("MultElem");
				MultScalarKernel_ = shader.FindKernel("MultScalar_");
				NegateKernel = shader.FindKernel("Negate");
				SubElemKernel = shader.FindKernel("SubElem");
				ZeroKernel_ = shader.FindKernel("Zero_");

            }
        }

		public void AbsGPU_() {
			if (dataOnGpu) {
				shader.SetBuffer (AbsKernel_, "abs_data_", dataBuffer);
				shader.Dispatch (AbsKernel_, this.size, 1, 1);
			}
		}

		public void AddScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.add_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(AddScalarKernel_, value, "add_scalar_scalar_");

				shader.SetBuffer(AddScalarKernel_, "add_scalar_data_", dataBuffer);
				shader.Dispatch(AddScalarKernel_, 1, 1, 1);

				valBuffer.Release();
			}
		}

		public void AddElemGPU_(FloatTensor tensor)
		{
			Debug.LogFormat("<color=blue>FloatTensor.add_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{

				shader.SetBuffer(AddScalarKernel_, "add_data_data_a_", dataBuffer);
				shader.SetBuffer(AddScalarKernel_, "add_data_data_b_", tensor.dataBuffer);
				shader.Dispatch(AddScalarKernel_, 1, 1, 1);

			}
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
			var result = new FloatTensor(shape, dataOnGpu);
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


        public FloatTensor MultElemGPU(FloatTensor other)
        {
            Debug.LogFormat("<color=blue>FloatTensor.elementwise_mult dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    var result = new FloatTensor(shape, dataOnGpu);
                    // correspond tensor buffers with shader kernel buffers
					shader.SetBuffer(MultElemKernel, "mult_elem_data_a", dataBuffer);
					shader.SetBuffer(MultElemKernel, "mult_elem_data_b", other.DataBuffer);
					shader.SetBuffer(MultElemKernel, "mult_elem_result", result.DataBuffer);

					shader.Dispatch(MultElemKernel, 1, 1, 1);
                    return result;
                }
            }
            else
            {
                Debug.Log("Tensors do not have the same number of elements!");
            }
            return this;
        }

		public FloatTensor MultScalarGPU_(float value)

		{
			Debug.LogFormat("<color=blue>FloatTensor.scalar_mult dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var scalarBuffer = SendFloatToGpu(MultScalarKernel_, value, "mult_scalar_scalar_");

				shader.SetBuffer(MultScalarKernel_, "mult_scalar_data_", dataBuffer);
				shader.Dispatch(MultScalarKernel_, 1, 1, 1);

				scalarBuffer.Release();

				return this; 
			}
			return this;
		}


        public FloatTensor NegateGPU()
        {
            if (dataOnGpu)
            {
                var result = new FloatTensor(shape, dataOnGpu);
				shader.SetBuffer(NegateKernel, "negate_data", dataBuffer);
				shader.SetBuffer(NegateKernel, "negate_result", result.dataBuffer);
				shader.Dispatch(NegateKernel, 1, 1, 1);
                return result;
            }
            return this;
        }

		public FloatTensor SubElemGPU(FloatTensor other)
		{
			//Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);

			if (size == other.Size)
			{
				if (dataOnGpu && other.DataOnGpu)
				{
					var result = new FloatTensor(shape, dataOnGpu);
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
