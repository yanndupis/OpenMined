using UnityEngine;
using System;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {

        public FloatTensor Add(FloatTensor x)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsAndShape(ref x);

            if (dataOnGpu)
            {
                // GPU Add Code Here
            }

            var result = new FloatTensor(shape, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    result.Data[i] = x.Data[i] + Data[i];
            });
            return result;
        }

        public FloatTensor AddMatrixMultiply(FloatTensor tensor1, FloatTensor tensor2)
        {
            // TODO: check for corner cases

            bool gpu = dataOnGpu & tensor1.DataOnGpu & tensor2.DataOnGpu;
            bool cpu = !(dataOnGpu | tensor1.DataOnGpu | tensor2.DataOnGpu);

            if (gpu)
            {
                AddMatrixMultiplyOnGpu(tensor1, tensor2);
            }
            else if (cpu)
            {
                //TODO: implement the function
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }
            return this;
        }

        public FloatTensor Ceil()
            // Returns a new Tensor with the smallest integer greater than or equal to each element
        {
            if (dataOnGpu)
            {
                return CeilOnGpu();
            }

            var result = new FloatTensor(shape, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    result.Data[i] = (float) (Math.Ceiling(Data[i]));
            });
            return result;
        }

        public FloatTensor ElementwiseSubtract(FloatTensor other)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);
            SameSizeDimensionsAndShape(ref other);

            if (dataOnGpu && other.DataOnGpu)
            {
                return ElementwiseSubtractOnGpu(other);
            }
            else if (!dataOnGpu && !other.dataOnGpu)
            {
                var result = new FloatTensor(shape, dataOnGpu);
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId =>
                {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                        result.Data[i] = Data[i] - other.Data[i];
                });
                return result;
            }

            throw new InvalidOperationException(
                "Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
        }

        public FloatTensor MulElementwise(FloatTensor other)
        {
            // Verify tensors are compatible for this operation
            SameSizeDimensionsAndShape(ref other);

            if (dataOnGpu && other.DataOnGpu)
            {
                return MulElementwiseGPU(other);
            }

            if (dataOnGpu || other.DataOnGpu)
                throw new InvalidOperationException(
                    "Data for both Tensors needs to be colocated on the same device. - CPU != GPU");

            var result = new FloatTensor(shape, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    result.Data[i] = data[i] * other.Data[i];
            });
            return result;
        }

        public FloatTensor MulScalar(float scalar)
        {
            if (dataOnGpu)
            {
                return MulScalarGPU(scalar);
            }

            var result = new FloatTensor(shape, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    result.Data[i] = Data[i] * scalar;
            });
            return result;
        }

        public FloatTensor MultiplyDerivative(FloatTensor other)
        {
            // TODO: check for corner cases
            if (dataOnGpu & other.DataOnGpu)
            {
                MultiplyDerivativeOnGpu(other);
            }
            else if (!dataOnGpu & !other.DataOnGpu)
            {
                //TODO: implement the function
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }
            return this;
        }

        public FloatTensor Neg()
        {
            if (dataOnGpu)
            {
                return NegGPU();
            }
            
            var result = new FloatTensor(shape, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = data.Length * (workerId + 1) / nCpu;
                for (var i = data.Length * workerId / nCpu; i < max; i++)
                    result[i] = -data[i];
            });
            return result;
        }

        public FloatTensor Transpose()
        {
            if (shape.Length != 2)
                throw new InvalidOperationException("Need to specify parameters for tensors with more than 2 dims.");

            return Transpose(0, 1);
        }

        public FloatTensor Transpose(int dimension1, int dimension2)
        {
            //TODO: Should we create a new Tensor object here?

            if (dimension1 < 0 || dimension1 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension1");
            if (dimension2 < 0 || dimension2 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension2");

            if (dimension1 == dimension2)
            {
                return this;
            }

            SwapElements(ref strides, dimension1, dimension2);
            SwapElements(ref shape, dimension1, dimension2);

            return this;
        }
    }
}