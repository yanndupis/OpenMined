using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private ComputeShader shader;

        [SerializeField] private int ScalarMultMain;
        private int ElementwiseMultMain;
        private int ElementwiseSubtractMain;
        private int SigmoidMatrixMultiply;
        private int MultiplyDerivative;
        private int AddMatrixMultiply;
        private int ResetWeights;

        public ComputeShader Shader
        {
            get { return shader; }
            set
            {
                shader = value;

                // save shaders and kernels
                ScalarMultMain = shader.FindKernel("ScalarMultMain");
                ElementwiseMultMain = shader.FindKernel("ElementwiseMultMain");
                ElementwiseSubtractMain = shader.FindKernel("ElementwiseSubtractMain");
                SigmoidMatrixMultiply = shader.FindKernel("SigmoidMatrixMultiply");
                MultiplyDerivative = shader.FindKernel("MultiplyDerivative");
                AddMatrixMultiply = shader.FindKernel("AddMatrixMultiply");
                ResetWeights = shader.FindKernel("ResetWeights");
            }
        }

        public void inline_elementwise_mult(FloatTensor other)
        {
            Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_mult dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    // correspond tensor buffers with shader kernel buffers
                    shader.SetBuffer(ElementwiseMultMain, "data_a", dataBuffer);
                    shader.SetBuffer(ElementwiseMultMain, "data_b", other.DataBuffer);

                    shader.Dispatch(ElementwiseMultMain, 1, 1, 1);
                }
                else if (!dataOnGpu && !other.DataOnGpu)
                {
                    for (int i = 0; i < size; i++)
                    {
                        data[i] = data[i] * other.data[i];
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

        public void scalar_mult(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.scalar_mult dataOnGpu: {0}</color>", dataOnGpu);

            if (dataOnGpu)
            {
                ComputeBuffer scalar_buffer = send_float_to_gpu(value, "temp_scalar");

                shader.SetBuffer(ScalarMultMain, "data", dataBuffer);
                shader.Dispatch(ScalarMultMain, 1, 1, 1);

                scalar_buffer.Release();
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    data[i] = data[i] * value;
                }
            }
        }

        public void inline_elementwise_subtract(FloatTensor other)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    // correspond tensor buffers with shader kernel buffers
                    shader.SetBuffer(ElementwiseSubtractMain, "data_c", dataBuffer);
                    shader.SetBuffer(ElementwiseSubtractMain, "data_d", other.DataBuffer);

                    shader.Dispatch(ElementwiseSubtractMain, size, 1, 1);
                }
                else if (!dataOnGpu && !other.dataOnGpu)
                {
                    for (int i = 0; i < size; i++)
                    {
                        data[i] = data[i] - other.data[i];
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

        public void init_sigmoid_matrix_multiply(FloatTensor tensor_1)
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

        public void multiply_derivative(FloatTensor tensor_1)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.multiply_derivative dataOnGpu: {0}</color>", dataOnGpu);
            shader.SetBuffer(MultiplyDerivative, "data_h", dataBuffer);
            shader.SetBuffer(MultiplyDerivative, "data_i", tensor_1.DataBuffer);
            shader.Dispatch(MultiplyDerivative, size, 1, 1);
        }

        public struct Dimensions
        {
            public int rows, columns;

            public Dimensions(int _rows, int _columns)
            {
                rows = _rows;
                columns = _columns;
            }

            public int stride()
            {
                return 2 * sizeof(int);
            }
        }

        public void init_add_matrix_multiply(FloatTensor tensor_1)
        {
            Dimensions[] dim = new Dimensions[]
            {
                new Dimensions(tensor_1.shape.Length, tensor_1.shape[0])
            };

            ComputeBuffer dim_buffer = new ComputeBuffer(dim.Length, dim[0].stride());
            dim_buffer.SetData(dim);
            shader.SetBuffer(AddMatrixMultiply, "dimensions_b", dim_buffer);
        }

        public void add_matrix_multiply(FloatTensor tensor_1, FloatTensor tensor_2)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
            shader.SetBuffer(AddMatrixMultiply, "data_j", dataBuffer);
            shader.SetBuffer(AddMatrixMultiply, "data_k", tensor_1.DataBuffer); //d
            shader.SetBuffer(AddMatrixMultiply, "data_l", tensor_2.DataBuffer);
            shader.Dispatch(AddMatrixMultiply, size, 1, 1);
        }

        public void init_weights(FloatTensor save_tensor)
        {
            shader.SetBuffer(ResetWeights, "weights", dataBuffer);
            shader.SetBuffer(ResetWeights, "original_weights", save_tensor.DataBuffer);
        }

        public void reset_weights()
        {
            shader.Dispatch(ResetWeights, size, 1, 1);
        }

        private ComputeBuffer send_float_to_gpu(float value, string name)
        {
            float[] scalar_array = new float[1];
            scalar_array[0] = value;

            ComputeBuffer scalar_buffer = new ComputeBuffer(1, sizeof(float));
            scalar_buffer.SetData(scalar_array);
            shader.SetBuffer(ScalarMultMain, name, scalar_buffer);

            return scalar_buffer;
        }
    }
}