using OpenMined.Syft.Shaders;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private ComputeShader shader;

        public FloatTensor AbsGPU(FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AbsGPU dataOnGpu: {0}</color>", dataOnGpu);
            if (!dataOnGpu) return result;
            var kernel = FloatTensorShader.AbsKernel;
            shader.SetBuffer(kernel, "abs_data", dataBuffer);
            shader.SetBuffer(kernel, "abs_result", result.dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            return result;
        }

        public void AbsGPU_()
        {
            Debug.LogFormat("<color=blue>FloatTensor.AbsGPU_ dataOnGpu: {0}</color>", dataOnGpu);
            if (!dataOnGpu) return;
            var kernel = FloatTensorShader.AbsKernel_;
            shader.SetBuffer(kernel, "abs_data_", dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
        }

        public void AddScalarGPU_(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AddScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            var kernel = FloatTensorShader.AddScalarKernel_;
            var valBuffer = SendFloatToGpu(kernel, value, "add_scalar_scalar_");
            shader.SetBuffer(kernel, "add_scalar_data_", dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            valBuffer.Release();
        }

        public void AddElemGPU_(FloatTensor tensor)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AddElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            if (this.id != tensor.id)
            {
                var kernel = FloatTensorShader.AddElemKernel_;
                shader.SetBuffer(kernel, "add_elem_data_a_", dataBuffer);
                shader.SetBuffer(kernel, "add_elem_data_b_", tensor.dataBuffer);
                shader.Dispatch(kernel, this.size, 1, 1);
            }
            else
            {
                Debug.LogFormat("addition with itself should be multiplication instead", dataOnGpu);
                this.MulScalarGPU_(2);
            }
        }

        public FloatTensor AddScalarGPU(float value, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AddScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            var kernel = FloatTensorShader.AddScalarKernel;
            var valBuffer = SendFloatToGpu(kernel, value, "add_scalar_scalar");
            shader.SetBuffer(kernel, "add_scalar_data", dataBuffer);
            shader.SetBuffer(kernel, "add_scalar_result", result.dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            valBuffer.Release();
            return result;
        }

        public FloatTensor AddElemGPU(FloatTensor tensor, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AddElemGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            // if tensors are not actually referring to the same tensor
            if (this.id != tensor.id)
            {
                var kernel = FloatTensorShader.AddElemKernel;
                shader.SetBuffer(kernel, "add_elem_data_a", this.DataBuffer);
                shader.SetBuffer(kernel, "add_elem_data_b", tensor.DataBuffer);
                shader.SetBuffer(kernel, "add_elem_data_result", result.DataBuffer);
                shader.Dispatch(kernel, this.size, 1, 1);
            }
            else
            {
                Debug.LogFormat("addition with itself should be multiplication instead", dataOnGpu);
                return this.MulScalarGPU(2, result);
            }
            return result;
        }

        public void AddMatrixMultiplyGPU(FloatTensor tensor_1, FloatTensor tensor_2)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
            var kernel = FloatTensorShader.AddMMKernel_;
            shader.SetBuffer(kernel, "addmm_data_a", dataBuffer);
            shader.SetBuffer(kernel, "addmm_data_b", tensor_1.DataBuffer); //d
            shader.SetBuffer(kernel, "addmm_data_c", tensor_2.DataBuffer);
            shader.Dispatch(kernel, size, 1, 1);
        }

        public void InitAddMatrixMultiplyGpu(FloatTensor tensor_1)
        {
            var dim = new Dimensions[]
            {
                new Dimensions(tensor_1.shape.Length, tensor_1.shape[0])
            };
            var kernel = FloatTensorShader.AddMMKernel_;
            var dimBuffer = new ComputeBuffer(dim.Length, dim[0].Stride());
            dimBuffer.SetData(dim);
            shader.SetBuffer(kernel, "addmm_dimensions", dimBuffer);
        }

        public FloatTensor CeilGPU()
        {
            Debug.LogFormat("<color=blue>FloatTensor.ceil dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return this;
            var result = new FloatTensor(shape, dataOnGpu);
            var kernel = FloatTensorShader.CeilKernel;
            shader.SetBuffer(kernel, "ceil_data", dataBuffer);
            shader.SetBuffer(kernel, "ceil_result", result.DataBuffer);
            shader.Dispatch(kernel, 1, 1, 1);
            return result;
        }

        public void FloorGPU_()
        {
            if (!DataOnGpu) return;
            var kernel = FloatTensorShader.FloorKernel_;
            shader.SetBuffer(kernel, "floor_data_", dataBuffer);
            shader.Dispatch(kernel, 1, 1, 1);
        }


        public void MulScalarGPU_(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            var kernel = FloatTensorShader.MulScalarKernel_;
            var valBuffer = SendFloatToGpu(kernel, value, "mul_scalar_scalar_");
            shader.SetBuffer(kernel, "mul_scalar_data_", dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            valBuffer.Release();
        }

        public void MulElemGPU_(FloatTensor tensor)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            if (tensor.id != this.id)
            {
                var kernel = FloatTensorShader.MulElemKernel_;
                shader.SetBuffer(kernel, "mul_elem_data_a_", dataBuffer);
                shader.SetBuffer(kernel, "mul_elem_data_b_", tensor.dataBuffer);
                shader.Dispatch(kernel, this.size, 1, 1);
            }
            else
            {
                PowGPU_(2);
            }
        }

        public FloatTensor MulScalarGPU(float value, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            var kernel = FloatTensorShader.MulScalarKernel;
            var valBuffer = SendFloatToGpu(kernel, value, "mul_scalar_scalar");
            shader.SetBuffer(kernel, "mul_scalar_data", dataBuffer);
            shader.SetBuffer(kernel, "mul_scalar_result", result.dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            valBuffer.Release();
            return result;
        }

        public FloatTensor MulElemGPU(FloatTensor tensor, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulElemGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            if (tensor.id != this.id)
            {
                var kernel = FloatTensorShader.MulElemKernel;
                shader.SetBuffer(kernel, "mul_elem_data_a", dataBuffer);
                shader.SetBuffer(kernel, "mul_elem_data_b", tensor.dataBuffer);
                shader.SetBuffer(kernel, "mul_elem_data_result", result.dataBuffer);
                shader.Dispatch(kernel, this.size, 1, 1);
            }
            else
            {
                return this.PowGPU(2, result);
            }
            return result;
        }


        public FloatTensor NegateGPU()
        {
            if (!dataOnGpu) return this;
            var result = new FloatTensor(shape, dataOnGpu);
            var kernel = FloatTensorShader.NegateKernel;
            shader.SetBuffer(kernel, "negate_data", dataBuffer);
            shader.SetBuffer(kernel, "negate_result", result.dataBuffer);
            shader.Dispatch(kernel, 1, 1, 1);
            return result;
        }

        public FloatTensor PowGPU(float value, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.PowGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            var kernel = FloatTensorShader.PowKernel;
            var valBuffer = SendFloatToGpu(kernel, value, "pow_scalar_scalar");
            shader.SetBuffer(kernel, "pow_scalar_data", dataBuffer);
            shader.SetBuffer(kernel, "pow_scalar_result", result.dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            valBuffer.Release();
            return result;
        }

        public void PowGPU_(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.PowGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            var kernel = FloatTensorShader.PowKernel_;
            var valBuffer = SendFloatToGpu(kernel, value, "pow_scalar_scalar_");
            shader.SetBuffer(kernel, "pow_scalar_data_", dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            valBuffer.Release();
        }

        public void SigmoidGPU_()
        {
            if (!dataOnGpu) return;
            var kernel = FloatTensorShader.SigmoidKernel_;
            shader.SetBuffer(kernel, "sigmoid_data_", dataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
        }


        public FloatTensor SubElemGPU(FloatTensor other)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    var result = new FloatTensor(shape, dataOnGpu);
                    var kernel = FloatTensorShader.SubElemKernel;
                    // correspond tensor buffers with shader kernel buffers
                    shader.SetBuffer(kernel, "sub_elem_data_a", dataBuffer);
                    shader.SetBuffer(kernel, "sub_elem_data_b", other.DataBuffer);
                    shader.SetBuffer(kernel, "sub_elem_result", result.DataBuffer);
                    shader.Dispatch(kernel, size, 1, 1);

                    return result;
                }
                Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
            }
            Debug.Log("Tensors do not have the same number of elements!");
            return this;
        }

        public FloatTensor TanhGPU()
        {
            var result = new FloatTensor(shape, dataOnGpu);
            var kernel = FloatTensorShader.TanhKernel;
            shader.SetBuffer(kernel, "tanh_data", dataBuffer);
            shader.SetBuffer(kernel, "tanh_result", result.DataBuffer);
            shader.Dispatch(kernel, this.size, 1, 1);
            return result;
        }

        public void ZeroGPU_()
        {
            var kernel = FloatTensorShader.ZeroKernel_;
            shader.SetBuffer(kernel, "zero_data_", dataBuffer);
            shader.Dispatch(kernel, 1, 1, 1);
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
            var scalarArray = new float[1];
            scalarArray[0] = value;

            var scalarBuffer = new ComputeBuffer(1, sizeof(float));
            scalarBuffer.SetData(scalarArray);
            shader.SetBuffer(kernel, name, scalarBuffer);

            return scalarBuffer;
        }
    }
}