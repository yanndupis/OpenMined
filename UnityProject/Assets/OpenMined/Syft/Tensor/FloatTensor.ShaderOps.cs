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
            shader.SetBuffer(FloatTensorShader.AbsKernel, "abs_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.AbsKernel, "abs_result", result.dataBuffer);
            shader.Dispatch(FloatTensorShader.AbsKernel, this.size, 1, 1);
            return result;
        }

        public void AbsGPU_()
        {
            Debug.LogFormat("<color=blue>FloatTensor.AbsGPU_ dataOnGpu: {0}</color>", dataOnGpu);
            if (!dataOnGpu) return;
            shader.SetBuffer(FloatTensorShader.AbsKernel_, "abs_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.AbsKernel_, this.size, 1, 1);
        }

        public void AddScalarGPU_(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AddScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            var valBuffer = SendFloatToGpu(FloatTensorShader.AddScalarKernel_, value, "add_scalar_scalar_");
            shader.SetBuffer(FloatTensorShader.AddScalarKernel_, "add_scalar_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.AddScalarKernel_, this.size, 1, 1);
            valBuffer.Release();
        }

        public void AddElemGPU_(FloatTensor tensor)
        {
            Debug.LogFormat("<color=blue>FloatTensor.AddElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            if (this.id != tensor.id)
            {
                shader.SetBuffer(FloatTensorShader.AddElemKernel_, "add_elem_data_a_", dataBuffer);
                shader.SetBuffer(FloatTensorShader.AddElemKernel_, "add_elem_data_b_", tensor.dataBuffer);
                shader.Dispatch(FloatTensorShader.AddElemKernel_, this.size, 1, 1);
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
            var valBuffer = SendFloatToGpu(FloatTensorShader.AddScalarKernel, value, "add_scalar_scalar");
            shader.SetBuffer(FloatTensorShader.AddScalarKernel, "add_scalar_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.AddScalarKernel, "add_scalar_result", result.dataBuffer);
            shader.Dispatch(FloatTensorShader.AddScalarKernel, this.size, 1, 1);
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
                shader.SetBuffer(FloatTensorShader.AddElemKernel, "add_elem_data_a", this.DataBuffer);
                shader.SetBuffer(FloatTensorShader.AddElemKernel, "add_elem_data_b", tensor.DataBuffer);
                shader.SetBuffer(FloatTensorShader.AddElemKernel, "add_elem_data_result", result.DataBuffer);
                shader.Dispatch(FloatTensorShader.AddElemKernel, this.size, 1, 1);
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
            shader.SetBuffer(FloatTensorShader.AddMMKernel_, "addmm_data_a", dataBuffer);
            shader.SetBuffer(FloatTensorShader.AddMMKernel_, "addmm_data_b", tensor_1.DataBuffer); //d
            shader.SetBuffer(FloatTensorShader.AddMMKernel_, "addmm_data_c", tensor_2.DataBuffer);
            shader.Dispatch(FloatTensorShader.AddMMKernel_, size, 1, 1);
        }

        public void InitAddMatrixMultiplyGpu(FloatTensor tensor_1)
        {
            var dim = new Dimensions[]
            {
                new Dimensions(tensor_1.shape.Length, tensor_1.shape[0])
            };

            var dimBuffer = new ComputeBuffer(dim.Length, dim[0].Stride());
            dimBuffer.SetData(dim);
            shader.SetBuffer(FloatTensorShader.AddMMKernel_, "addmm_dimensions", dimBuffer);
        }

        public FloatTensor CeilGPU()
        {
            Debug.LogFormat("<color=blue>FloatTensor.ceil dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return this;
            var result = new FloatTensor(shape, dataOnGpu);
            shader.SetBuffer(FloatTensorShader.CeilKernel, "ceil_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.CeilKernel, "ceil_result", result.DataBuffer);
            shader.Dispatch(FloatTensorShader.CeilKernel, 1, 1, 1);
            return result;
        }

        public void FloorGPU_()
        {
            if (!DataOnGpu) return;
            shader.SetBuffer(FloatTensorShader.FloorKernel_, "floor_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.FloorKernel_, 1, 1, 1);
        }


        public void MulScalarGPU_(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            var valBuffer = SendFloatToGpu(FloatTensorShader.MulScalarKernel_, value, "mul_scalar_scalar_");
            shader.SetBuffer(FloatTensorShader.MulScalarKernel_, "mul_scalar_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.MulScalarKernel_, this.size, 1, 1);
            valBuffer.Release();
        }

        public void MulElemGPU_(FloatTensor tensor)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            if (tensor.id != this.id)
            {
                shader.SetBuffer(FloatTensorShader.MulElemKernel_, "mul_elem_data_a_", dataBuffer);
                shader.SetBuffer(FloatTensorShader.MulElemKernel_, "mul_elem_data_b_", tensor.dataBuffer);
                shader.Dispatch(FloatTensorShader.MulElemKernel_, this.size, 1, 1);
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
            var valBuffer = SendFloatToGpu(FloatTensorShader.MulScalarKernel, value, "mul_scalar_scalar");
            shader.SetBuffer(FloatTensorShader.MulScalarKernel, "mul_scalar_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.MulScalarKernel, "mul_scalar_result", result.dataBuffer);
            shader.Dispatch(FloatTensorShader.MulScalarKernel, this.size, 1, 1);
            valBuffer.Release();
            return result;
        }

        public FloatTensor MulElemGPU(FloatTensor tensor, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.MulElemGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            if (tensor.id != this.id)
            {
                shader.SetBuffer(FloatTensorShader.MulElemKernel, "mul_elem_data_a", dataBuffer);
                shader.SetBuffer(FloatTensorShader.MulElemKernel, "mul_elem_data_b", tensor.dataBuffer);
                shader.SetBuffer(FloatTensorShader.MulElemKernel, "mul_elem_data_result", result.dataBuffer);
                shader.Dispatch(FloatTensorShader.MulElemKernel, this.size, 1, 1);
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
            shader.SetBuffer(FloatTensorShader.NegateKernel, "negate_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.NegateKernel, "negate_result", result.dataBuffer);
            shader.Dispatch(FloatTensorShader.NegateKernel, 1, 1, 1);
            return result;
        }

        public FloatTensor PowGPU(float value, FloatTensor result)
        {
            Debug.LogFormat("<color=blue>FloatTensor.PowGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return result;
            var valBuffer = SendFloatToGpu(FloatTensorShader.PowKernel, value, "pow_scalar_scalar");
            shader.SetBuffer(FloatTensorShader.PowKernel, "pow_scalar_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.PowKernel, "pow_scalar_result", result.dataBuffer);
            shader.Dispatch(FloatTensorShader.PowKernel, this.size, 1, 1);
            valBuffer.Release();
            return result;
        }

        public void PowGPU_(float value)
        {
            Debug.LogFormat("<color=blue>FloatTensor.PowGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (!dataOnGpu) return;
            var valBuffer = SendFloatToGpu(FloatTensorShader.PowKernel_, value, "pow_scalar_scalar_");
            shader.SetBuffer(FloatTensorShader.PowKernel_, "pow_scalar_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.PowKernel_, this.size, 1, 1);
            valBuffer.Release();
        }

        public void SigmoidGPU_()
        {
            if (!dataOnGpu) return;
            shader.SetBuffer(FloatTensorShader.SigmoidKernel_, "sigmoid_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.SigmoidKernel_, this.size, 1, 1);
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
                    shader.SetBuffer(FloatTensorShader.SubElemKernel, "sub_elem_data_a", dataBuffer);
                    shader.SetBuffer(FloatTensorShader.SubElemKernel, "sub_elem_data_b", other.DataBuffer);
                    shader.SetBuffer(FloatTensorShader.SubElemKernel, "sub_elem_result", result.DataBuffer);
                    shader.Dispatch(FloatTensorShader.SubElemKernel, size, 1, 1);

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
            shader.SetBuffer(FloatTensorShader.TanhKernel, "tanh_data", dataBuffer);
            shader.SetBuffer(FloatTensorShader.TanhKernel, "tanh_result", result.DataBuffer);
            shader.Dispatch(FloatTensorShader.TanhKernel, this.size, 1, 1);
            return result;
        }

        public void ZeroGPU_()
        {
            shader.SetBuffer(FloatTensorShader.ZeroKernel_, "zero_data_", dataBuffer);
            shader.Dispatch(FloatTensorShader.ZeroKernel_, 1, 1, 1);
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