using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private ComputeShader shader;

        [SerializeField] 
        private static int ScalarMultMainKernel;
        private static int ElementwiseMultMainKernel;
        private static int ElementwiseSubtractMainKernel;
        private static int MultiplyDerivativeKernel;
        private static int AddMatrixMultiplyKernel;
		private static int NegateValuesKernel;
		private static int ZeroValuesKernel;

        public ComputeShader Shader
        {
            get { return shader; }
            set
            {
                shader = value;

                // save shaders and kernels
                ScalarMultMainKernel = shader.FindKernel("ScalarMultMain");
                ElementwiseMultMainKernel = shader.FindKernel("ElementwiseMultMain");
                ElementwiseSubtractMainKernel = shader.FindKernel("ElementwiseSubtractMain");
                MultiplyDerivativeKernel = shader.FindKernel("MultiplyDerivative");
                AddMatrixMultiplyKernel = shader.FindKernel("AddMatrixMultiply");
				NegateValuesKernel = shader.FindKernel ("NegateValues");
            }
        }

		public void NegGPU() {

			shader.SetBuffer (NegateValuesKernel, "data_neg", dataBuffer);
			shader.Dispatch (NegateValuesKernel, 1, 1, 1);

		}

		public void ZeroGPU_() {

			shader.SetBuffer (ZeroValuesKernel, "data_zero_", dataBuffer);
			shader.Dispatch (ZeroValuesKernel, 1, 1, 1);

		}

        public void ElementwiseMultiplicationOnGpu(FloatTensor other)
        {
            Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_mult dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    // correspond tensor buffers with shader kernel buffers
                    shader.SetBuffer(ElementwiseMultMainKernel, "data_a", dataBuffer);
                    shader.SetBuffer(ElementwiseMultMainKernel, "data_b", other.DataBuffer);

                    shader.Dispatch(ElementwiseMultMainKernel, 1, 1, 1);
                }
            }
            else
            {
                Debug.Log("Tensors do not have the same number of elements!");
            }
        }

        public void ScalarMultiplicationOnGpu(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.scalar_mult dataOnGpu: {0}</color>", dataOnGpu);

            if (dataOnGpu)
            {
                ComputeBuffer scalarBuffer = SendFloatToGpu(value, "temp_scalar");

                shader.SetBuffer(ScalarMultMainKernel, "data", dataBuffer);
                shader.Dispatch(ScalarMultMainKernel, 1, 1, 1);

                scalarBuffer.Release();
            }
        }

        public void ElementwiseSubtractOnGpu(FloatTensor other)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    // correspond tensor buffers with shader kernel buffers
                    shader.SetBuffer(ElementwiseSubtractMainKernel, "data_c", dataBuffer);
                    shader.SetBuffer(ElementwiseSubtractMainKernel, "data_d", other.DataBuffer);

                    shader.Dispatch(ElementwiseSubtractMainKernel, size, 1, 1);
                }
                else if (!dataOnGpu && !other.dataOnGpu)
                {
                    for (int i = 0; i < size; i++)
                    {
                        Data[i] = Data[i] - other.Data[i];
                    }
                }
                else
                {
                    Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
                }
            }
            else
            {
                Debug.Log("Tensors do not have the same number of elements!");
            }
        }

        // TODO we should split these functions. i.e. sigmoid_matrix_multiply -> sigmoid and multiply 
       /* public void init_sigmoid_matrix_multiply(FloatTensor tensor_1)
        {
            Dimensions[] dim = new Dimensions[]
            {
                new Dimensions(tensor_1.Shape.Length, tensor_1.Shape[0])
            };

            ComputeBuffer dim_buffer = new ComputeBuffer(dim.Length, dim[0].stride());
            dim_buffer.SetData(dim);
            shader.SetBuffer(SigmoidMatrixMultiply, "dimensions_a", dim_buffer);
        }

        public void sigmoid_matrix_multiply(FloatTensor tensor_1, FloatTensor tensor_2)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.sigmoid_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
            shader.SetBuffer(SigmoidMatrixMultiply, "data_e", dataBuffer);
            shader.SetBuffer(SigmoidMatrixMultiply, "data_f", tensor_1.dataBuffer);
            shader.SetBuffer(SigmoidMatrixMultiply, "data_g", tensor_2.dataBuffer);
            shader.Dispatch(SigmoidMatrixMultiply, size, 1, 1);
        }
*/
        public void MultiplyDerivativeOnGpu(FloatTensor tensor_1)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.multiply_derivative dataOnGpu: {0}</color>", dataOnGpu);
            shader.SetBuffer(MultiplyDerivativeKernel, "data_h", dataBuffer);
            shader.SetBuffer(MultiplyDerivativeKernel, "data_i", tensor_1.DataBuffer);
            shader.Dispatch(MultiplyDerivativeKernel, size, 1, 1);
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

        public void InitAddMatrixMultiplyOnGpu(FloatTensor tensor_1)
        {
            Dimensions[] dim = new Dimensions[]
            {
                new Dimensions(tensor_1.shape.Length, tensor_1.shape[0])
            };

            ComputeBuffer dimBuffer = new ComputeBuffer(dim.Length, dim[0].Stride());
            dimBuffer.SetData(dim);
            shader.SetBuffer(AddMatrixMultiplyKernel, "dimensions_b", dimBuffer);
        }

        public void AddMatrixMultiplyOnGpu(FloatTensor tensor_1, FloatTensor tensor_2)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
            shader.SetBuffer(AddMatrixMultiplyKernel, "data_j", dataBuffer);
            shader.SetBuffer(AddMatrixMultiplyKernel, "data_k", tensor_1.DataBuffer); //d
            shader.SetBuffer(AddMatrixMultiplyKernel, "data_l", tensor_2.DataBuffer);
            shader.Dispatch(AddMatrixMultiplyKernel, size, 1, 1);
        }

        private ComputeBuffer SendFloatToGpu(float value, string name)
        {
            float[] scalarArray = new float[1];
            scalarArray[0] = value;

            ComputeBuffer scalarBuffer = new ComputeBuffer(1, sizeof(float));
            scalarBuffer.SetData(scalarArray);
            shader.SetBuffer(ScalarMultMainKernel, name, scalarBuffer);

            return scalarBuffer;
        }
    }
}