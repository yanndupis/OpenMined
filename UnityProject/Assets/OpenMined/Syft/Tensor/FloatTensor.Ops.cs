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
            bool gpu = dataOnGpu & tensor1.DataOnGpu & tensor2.DataOnGpu;
            bool cpu = !(dataOnGpu | tensor1.DataOnGpu | tensor2.DataOnGpu);

            int[] res_shape = this.Shape;
            int[] shape1 = tensor1.Shape;
            int[] shape2 = tensor2.Shape;

            if (shape1[1] != shape2[0])
                throw new InvalidOperationException(String.Format("Matrix multiply not possible: {0} & {1}.", shape1[1], shape2[0]));
            if (res_shape[0] != shape1[0])
                throw new InvalidOperationException(String.Format("First dimension doesn't match: {0} vs {1}.", res_shape[0], shape1[0]));
            if (res_shape[1] != shape2[1])
                throw new InvalidOperationException(String.Format("Last dimension doesn't match: {0} vs {1}.", res_shape[res_shape.Length - 1],shape2[shape2.Length - 1]));

            if (gpu)
            {
                AddMatrixMultiplyGPU(tensor1, tensor2);
            }
            else if (cpu)
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId =>
                {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                    {
                        int actual2 = i % res_shape[1];
                        int actual1 = (i - actual2) / res_shape[1];
                        for (var j = 0; j < shape1[1]; j++)
                        {
                            Data[i] += tensor1.Data[j + actual1 * shape1[1]] * tensor2.Data[j * shape2[1] + actual2];
                        }
                    }
                });
                return this;
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
				return CeilGPU();
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

       

        public FloatTensor MulElementwise(FloatTensor other)
        {
            // Verify tensors are compatible for this operation
            SameSizeDimensionsAndShape(ref other);

            if (dataOnGpu && other.DataOnGpu)
            {
				return MultElemGPU(other);
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
				return MultScalarGPU_(scalar);
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

        public FloatTensor Neg()
        {
            if (dataOnGpu)
            {
				return NegateGPU();
            }
            
            var result = new FloatTensor(shape, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = data.Length * (workerId + 1) / nCpu;
                for (var i = data.Length * workerId / nCpu; i < max; i++)
                    result.data[i] = -data[i];
            });
            return result;
        }

		public FloatTensor SubtractElementwise(FloatTensor other)
		{
			//Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);
			SameSizeDimensionsAndShape(ref other);

			if (dataOnGpu && other.DataOnGpu)
			{
				return SubElemGPU(other);
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