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
		private static int DivScalarKernel_;
		[SerializeField]
		private static int DivElemKernel_;
		[SerializeField]
		private static int DivScalarKernel;
		[SerializeField]
		private static int DivElemKernel;
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
	    private static int SqrtKernel;
		[SerializeField]
		private static int SubScalarKernel_;
		[SerializeField]
		private static int SubElemKernel_;
		[SerializeField]
		private static int SubScalarKernel;
		[SerializeField]
		private static int SubElemKernel;
		[SerializeField]
		private static int TanhKernel;
   	 	[SerializeField]
		private static int TriuKernel_;
    	[SerializeField]
    	private static int TruncKernel;
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
				DivScalarKernel_ = shader.FindKernel ("DivScalar_");
				DivElemKernel_ = shader.FindKernel ("DivElem_");
				DivScalarKernel = shader.FindKernel ("DivScalar");
				DivElemKernel = shader.FindKernel ("DivElem");
				FloorKernel_ = shader.FindKernel ("Floor_");
				MulScalarKernel_ = shader.FindKernel ("MulScalar_");
				MulElemKernel_ = shader.FindKernel ("MulElem_");
				MulScalarKernel = shader.FindKernel ("MulScalar");
				MulElemKernel = shader.FindKernel ("MulElem");
				NegateKernel = shader.FindKernel ("Negate");
				PowKernel = shader.FindKernel ("Pow");
				PowKernel_ = shader.FindKernel ("Pow_");
				SigmoidKernel_ = shader.FindKernel ("Sigmoid_");
				SqrtKernel = shader.FindKernel("Sqrt");
				SubScalarKernel_ = shader.FindKernel ("SubScalar_");
				SubElemKernel_ = shader.FindKernel ("SubElem_");
				SubScalarKernel = shader.FindKernel ("SubScalar");
				SubElemKernel = shader.FindKernel ("SubElem");
				TanhKernel = shader.FindKernel ("Tanh");
        		TriuKernel_ = shader.FindKernel ("Triu_");
       			TruncKernel = shader.FindKernel ("Trunc");
				ZeroKernel_ = shader.FindKernel ("Zero_");
			}

		}

		public FloatTensor AbsGPU(FloatTensor result) {
			Debug.LogFormat("<color=blue>FloatTensor.AbsGPU dataOnGpu: {0}</color>", dataOnGpu);
			if (dataOnGpu) {
				shader.SetBuffer (AbsKernel, "AbsData", dataBuffer);
				shader.SetBuffer (AbsKernel, "AbsResult", result.dataBuffer);
				shader.Dispatch (AbsKernel, this.size, 1, 1);
			}
			return result;
		}

		public void AbsGPU_() {
			Debug.LogFormat("<color=blue>FloatTensor.AbsGPU_ dataOnGpu: {0}</color>", dataOnGpu);
			if (dataOnGpu) {
				shader.SetBuffer (AbsKernel_, "AbsData_", dataBuffer);
				shader.Dispatch (AbsKernel_, this.size, 1, 1);
			}
		}

		public void AddScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.AddScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(AddScalarKernel_, value, "AddScalarScalar_");

				shader.SetBuffer(AddScalarKernel_, "AddScalarData_", dataBuffer);
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
					shader.SetBuffer (AddElemKernel_, "AddElemDataA_", dataBuffer);
					shader.SetBuffer (AddElemKernel_, "AddElemDataB_", tensor.dataBuffer);
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
				var valBuffer = SendFloatToGpu(AddScalarKernel, value, "AddScalarScalar");

				shader.SetBuffer(AddScalarKernel, "AddScalarData", dataBuffer);
				shader.SetBuffer(AddScalarKernel, "AddScalarResult", result.dataBuffer);
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
					shader.SetBuffer (AddElemKernel, "AddElemDataA", this.DataBuffer);
					shader.SetBuffer (AddElemKernel, "AddElemDataB", tensor.DataBuffer);
					shader.SetBuffer (AddElemKernel, "AddElemDataResult", result.DataBuffer);
					shader.Dispatch (AddElemKernel, this.size, 1, 1);
				} else {
					Debug.LogFormat("addition with itself should be multiplication instead", dataOnGpu);
					return this.MulScalarGPU (2, result);
				}


			}
			return result;
		}

		public void DivScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.DivScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(DivScalarKernel_, value, "DivScalarScalar_");

				shader.SetBuffer(DivScalarKernel_, "DivScalarData_", dataBuffer);
				shader.Dispatch(DivScalarKernel_, this.size, 1, 1);

				valBuffer.Release();
			}
		}

		public void DivElemGPU_(FloatTensor tensor)
		{
			Debug.LogFormat("<color=blue>FloatTensor.DivElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				if (tensor.id != this.id) {
					shader.SetBuffer (DivElemKernel_, "DivElemDataA_", dataBuffer);
					shader.SetBuffer (DivElemKernel_, "DivElemDataB_", tensor.dataBuffer);
					shader.Dispatch (DivElemKernel_, this.size, 1, 1);
				} else {
					tensor.Zero_();
					tensor.Add_(1);
				}

			}
		}

		public FloatTensor DivScalarGPU(float value, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.DivScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(DivScalarKernel, value, "DivScalarScalar");

				shader.SetBuffer(DivScalarKernel, "DivScalarData", dataBuffer);
				shader.SetBuffer(DivScalarKernel, "DivScalarResult", result.dataBuffer);
				shader.Dispatch(DivScalarKernel, this.size, 1, 1);

				valBuffer.Release();
			}
			return result;
		}

		public FloatTensor DivElemGPU(FloatTensor tensor, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.DivElemGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				if (tensor.id != this.id) {
					shader.SetBuffer (DivElemKernel, "DivElemDataA", dataBuffer);
					shader.SetBuffer (DivElemKernel, "DivElemDataB", tensor.dataBuffer);
					shader.SetBuffer (DivElemKernel, "DivElemDataResult", result.dataBuffer);
					shader.Dispatch (DivElemKernel, this.size, 1, 1);
				} else {
					result.Add_ (1);
					return result;
				}

			}
			return result;
		}


		public void AddMatrixMultiplyGPU(FloatTensor tensor_1, FloatTensor tensor_2)
		{
			//Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
			shader.SetBuffer(AddMMKernel_, "AddmmDataA", dataBuffer);
			shader.SetBuffer(AddMMKernel_, "AddmmDataB", tensor_1.DataBuffer); //d
			shader.SetBuffer(AddMMKernel_, "AddmmDataC", tensor_2.DataBuffer);
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
			shader.SetBuffer(AddMMKernel_, "AddmmDimensions", dimBuffer);
		}

		public FloatTensor CeilGPU()
		{
			Debug.LogFormat("<color=blue>FloatTensor.ceil dataOnGpu: {0}</color>", dataOnGpu);

			if (!dataOnGpu) return this;
			var result = new FloatTensor(shape, this.shader, dataOnGpu);
			shader.SetBuffer(CeilKernel, "CeilData", dataBuffer);
			shader.SetBuffer(CeilKernel, "CeilResult", result.DataBuffer);
			shader.Dispatch(CeilKernel, 1, 1, 1);
			return result;
		}

        	public void FloorGPU_()
        	{
            		if (DataOnGpu)
            		{
                		shader.SetBuffer(FloorKernel_, "FloorData_", dataBuffer);
                		shader.Dispatch(FloorKernel_, 1, 1, 1);
            		}
        	}


		public void MulScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.MulScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(MulScalarKernel_, value, "MulScalarScalar_");

				shader.SetBuffer(MulScalarKernel_, "MulScalarData_", dataBuffer);
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
					shader.SetBuffer (MulElemKernel_, "MulElemDataA_", dataBuffer);
					shader.SetBuffer (MulElemKernel_, "MulElemDataB_", tensor.dataBuffer);
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
				var valBuffer = SendFloatToGpu(MulScalarKernel, value, "MulScalarScalar");

				shader.SetBuffer(MulScalarKernel, "MulScalarData", dataBuffer);
				shader.SetBuffer(MulScalarKernel, "MulScalarResult", result.dataBuffer);
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
					shader.SetBuffer (MulElemKernel, "MulElemDataA", dataBuffer);
					shader.SetBuffer (MulElemKernel, "MulElemDataB", tensor.dataBuffer);
					shader.SetBuffer (MulElemKernel, "MulElemDataResult", result.dataBuffer);
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
				shader.SetBuffer(NegateKernel, "NegateData", dataBuffer);
				shader.SetBuffer(NegateKernel, "NegateResult", result.dataBuffer);
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
				var valBuffer = SendFloatToGpu(PowKernel, value, "PowScalarScalar");

				shader.SetBuffer(PowKernel, "PowScalarData", dataBuffer);
				shader.SetBuffer(PowKernel, "PowScalarResult", result.dataBuffer);
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
				var valBuffer = SendFloatToGpu(PowKernel_, value, "PowScalarScalar_");

				shader.SetBuffer(PowKernel_, "PowScalarData_", dataBuffer);
				shader.Dispatch(PowKernel_, this.size, 1, 1);

				valBuffer.Release();
			}
		}

	    private FloatTensor SqrtGPU()
	    {
		    if (!dataOnGpu) return this;
		    
		    var result = new FloatTensor(shape, shader, dataOnGpu);
		    shader.SetBuffer(SqrtKernel, "SqrtData", dataBuffer);
		    shader.SetBuffer(SqrtKernel, "SqrtResult", result.dataBuffer);
		    shader.Dispatch(SqrtKernel, size, 1, 1);
		    
		    return result;
	    }

        public void SigmoidGPU_()
        {
            if (dataOnGpu)
            {
                shader.SetBuffer(SigmoidKernel_, "SigmoidData_", dataBuffer);
                shader.Dispatch(SigmoidKernel_, this.size, 1, 1);
            }
        }


		public void SubScalarGPU_(float value)
		{
			Debug.LogFormat("<color=blue>FloatTensor.SubScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(SubScalarKernel_, value, "SubScalarScalar_");

				shader.SetBuffer(SubScalarKernel_, "SubScalarData_", dataBuffer);
				shader.Dispatch(SubScalarKernel_, this.size, 1, 1);

				valBuffer.Release();
			}
		}

		public void SubElemGPU_(FloatTensor tensor)
		{
			Debug.LogFormat("<color=blue>FloatTensor.SubElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				if (this.id != tensor.id) {
					shader.SetBuffer (SubElemKernel_, "SubElemDataA_", dataBuffer);
					shader.SetBuffer (SubElemKernel_, "SubElemDataB_", tensor.dataBuffer);
					shader.Dispatch (SubElemKernel_, this.size, 1, 1);
				} else {
					Debug.LogFormat("addition with itself should be multiplication instead", dataOnGpu);
					this.Zero_ ();
				}

			}
		}

		public FloatTensor SubScalarGPU(float value, FloatTensor result)
		{
			Debug.LogFormat("<color=blue>FloatTensor.SubScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				var valBuffer = SendFloatToGpu(SubScalarKernel, value, "SubScalarScalar");

				shader.SetBuffer(SubScalarKernel, "SubScalarData", dataBuffer);
				shader.SetBuffer(SubScalarKernel, "SubScalarResult", result.dataBuffer);
				shader.Dispatch(SubScalarKernel, this.size, 1, 1);

				valBuffer.Release();
			}
			return result;
		}

		public FloatTensor SubElemGPU(FloatTensor tensor, FloatTensor result)
		{

			Debug.LogFormat("<color=blue>FloatTensor.SubElemGPU dataOnGpu: {0}</color>", dataOnGpu);

			if (dataOnGpu)
			{
				// if tensors are not actually referring to the same tensor
				if (this.id != tensor.id) {
					shader.SetBuffer (SubElemKernel, "SubElemDataA", this.DataBuffer);
					shader.SetBuffer (SubElemKernel, "SubElemDataB", tensor.DataBuffer);
					shader.SetBuffer (SubElemKernel, "SubElemDataResult", result.DataBuffer);
					shader.Dispatch (SubElemKernel, this.size, 1, 1);
				} else {
					// should return a tensor of zeros.
					return result;
				}


			}
			return result;
		}

		public FloatTensor TanhGPU ()
		{
			var result = new FloatTensor(shape, this.shader, dataOnGpu);
			shader.SetBuffer(TanhKernel, "TanhData", dataBuffer);
			shader.SetBuffer(TanhKernel, "TanhResult", result.DataBuffer);
			shader.Dispatch(TanhKernel, this.size, 1, 1);
			return result;
		}

    public void TriuGPU_(int k)
    {
      var dim = new Dimensions[]
      {
        new Dimensions(this.shape[0], this.shape[1])
      };

      var dimBuffer = new ComputeBuffer(2, dim[0].Stride());
      var kBuffer = SendIntToGpu(TriuKernel_, k, "TriuK_");
      dimBuffer.SetData(dim);
      shader.SetBuffer(TriuKernel_, "TriuDimensions_", dimBuffer);
      shader.SetBuffer(TriuKernel_, "TriuData_", dataBuffer);
      shader.Dispatch(TriuKernel_, this.size, 1, 1);

      dimBuffer.Release();
      kBuffer.Release();

    }

        public FloatTensor TruncGPU ()
        {
            var result = new FloatTensor(shape, this.shader, dataOnGpu);
            shader.SetBuffer(TruncKernel, "TruncData", dataBuffer);
            shader.SetBuffer(TruncKernel, "TruncResult", result.DataBuffer);
            shader.Dispatch(TruncKernel, this.size, 1, 1);
            return result;
        }


        public void ZeroGPU_()
        {
			shader.SetBuffer(ZeroKernel_, "ZeroData_", dataBuffer);
			shader.Dispatch(ZeroKernel_, this.size, 1, 1);
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

        private ComputeBuffer SendIntToGpu(int kernel, int value, string name)
        {
            int[] array = new int[1];
            array[0] = value;

            var arrayBuffer = new ComputeBuffer(1, sizeof(float));
            arrayBuffer.SetData(array);
            shader.SetBuffer(kernel, name, arrayBuffer);

            return arrayBuffer;
        }
    }
}
