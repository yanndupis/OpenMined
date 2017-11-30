using UnityEngine;
using System;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {

		public FloatTensor Abs()
		// Returns a new Tensor with the smallest integer greater than or equal to each element
		{

			var result = new FloatTensor(shape, this.shader, dataOnGpu);

			if (dataOnGpu) {

				result.Gpu ();
				return AbsGPU (result);

			} else {

				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = (float)(Math.Abs (Data [i]));
				});
			}
			return result;
		}


		public FloatTensor Add(FloatTensor x)
        {
            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

			FloatTensor result = new FloatTensor (shape, this.shader);

			if (dataOnGpu & x.dataOnGpu) {
				result.Gpu ();
				AddElemGPU (x, result);

			} else {


				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = x.Data [i] + Data [i];
				});

			}

			return result;
        }

		public FloatTensor Add(float value)
		{
			var result = new FloatTensor(shape, this.shader, false);

			if (dataOnGpu) {

				result.Gpu ();
				return AddScalarGPU (value, result);

			} else {

				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = value + Data [i];
				});
			}
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
                    for (var idx = size * workerId / nCpu; idx < max; idx++)
                    {
                        int col = idx % res_shape[1];
                        int row = (idx - col) / res_shape[1];
                        int row_offset = row * shape1[1];
                        for (var j = 0; j < shape1[1]; j++)
                        {
                            Data[idx] += tensor1.Data[j + row_offset] * tensor2.Data[j * shape2[1] + col];
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

			var result = new FloatTensor(shape, this.shader, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    result.Data[i] = (float) (Math.Ceiling(Data[i]));
            });
            return result;
        }

		public FloatTensor Div(FloatTensor x)
		{
			// Check if both tensors are compatible for sum
			SameSizeDimensionsShapeAndLocation(ref x);

			var result = new FloatTensor (shape, this.shader, false);

			if (dataOnGpu) {

				result.Gpu ();
				return DivElemGPU (x, result);

			} else {


				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = Data [i] / x.Data [i];
				});

			}
			return result;
		}

		public FloatTensor Div(float value)
		{
			var result = new FloatTensor(shape, this.shader, false);

			if (dataOnGpu) {
				result.Gpu ();
				return DivScalarGPU (value, result);
			} else {

				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = Data [i] / value;
				});
			}
			return result;
		}



		public FloatTensor Mul(FloatTensor x)
		{
			// Check if both tensors are compatible for sum
			SameSizeDimensionsShapeAndLocation(ref x);

			var result = new FloatTensor (shape, this.shader, false);

			if (dataOnGpu) {

				result.Gpu ();
				return MulElemGPU (x, result);

			} else {


				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = x.Data [i] * Data [i];
				});

			}
			return result;
		}

		public FloatTensor Mul(float value)
		{
			var result = new FloatTensor(shape, this.shader, false);

			if (dataOnGpu) {
				result.Gpu ();
				return MulScalarGPU (value, result);
			} else {

				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = value * Data [i];
				});
			}
			return result;
		}

        public FloatTensor Neg()
        {
            if (dataOnGpu)
            {
				return NegateGPU();
            }

			var result = new FloatTensor(shape, this.shader, dataOnGpu);
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = data.Length * (workerId + 1) / nCpu;
                for (var i = data.Length * workerId / nCpu; i < max; i++)
                    result.data[i] = -data[i];
            });
            return result;
        }

		public FloatTensor Sub(FloatTensor x)
		{
			// Check if both tensors are compatible for sum
			SameSizeDimensionsShapeAndLocation(ref x);

			FloatTensor result = new FloatTensor (shape, this.shader);

			if (dataOnGpu & x.dataOnGpu) {
				result.Gpu ();
				SubElemGPU (x, result);

			} else {


				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = x.Data [i] - Data [i];
				});

			}

			return result;
		}

		public FloatTensor Sub(float value)
		{
			var result = new FloatTensor(shape, this.shader, false);

			if (dataOnGpu) {

				result.Gpu ();
				return SubScalarGPU (value, result);

			} else {

				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						result.Data [i] = value - Data [i];
				});
			}
			return result;
		}

		public FloatTensor Sum(int dim)
		{
			int[] result_shape = new int[shape.Length - 1];
			int j = 0;
			for (var i = 0; i < shape.Length; i++) {
				if (i != dim) {
					result_shape [j] = shape [i];
					j += 1;
				}
			}

			var result = new FloatTensor(result_shape, this.shader, false);


			if (dataOnGpu) {
				// TODO: write GPU kernel for summing over a dimension
//				result.Gpu ();

			} else {



				// TODO: write parallel.for for summing over a dimension
//				var nCpu = SystemInfo.processorCount;
//				Parallel.For (0, nCpu, workerId => {
//					var max = size * (workerId + 1) / nCpu;
//					for (var i = size * workerId / nCpu; i < max; i++)
//						result.Data [i] = value - Data [i];
//				});
			}
			return result;
		}

        public FloatTensor Tanh()
        {
            if (dataOnGpu)
            {
                return TanhGPU();
            }
            else
            {
                var result = new FloatTensor(shape, this.shader, dataOnGpu);
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId =>
                {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                    {
                        var d = (double) Data[i];
                        result.Data[i] = (float) System.Math.Tanh(d);
                    }
                });

                return result;
            }
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

        public void Triu_(int k)
        {
            if (shape.Length != 2)
            {
              throw new InvalidOperationException(String.Format("Matrix multiply not possible: Num. Dimensions {0} != 2.", shape.Length));
            }
            if (dataOnGpu)
            {
              UnityEngine.Debug.Log("Entra");
                TriuGPU_(k);
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    int col = i % this.shape[1];
                    int row = (i - col) / this.shape[1];
                    if (col < row + k)
                    {
                      Data[i] = 0.0f;
                    }
                }
            });
        }
    }
}
