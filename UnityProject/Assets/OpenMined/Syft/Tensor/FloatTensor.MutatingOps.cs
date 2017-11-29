using System;
using System.Threading.Tasks;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor

    {

		public void Abs_()
		{
			if (dataOnGpu)
			{
				AbsGPU_ ();
				return;
			}
			var nCpu = SystemInfo.processorCount;
			Parallel.For(0, nCpu, workerId =>
				{
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						if (data[i] < 0)
							data[i] = -data[i];
				});
		}

		public void Add_(FloatTensor x)
		{
			SameSizeDimensionsShapeAndLocation(ref x);

			if (dataOnGpu) {
				AddElemGPU_ (x);

			} else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						data [i] += x.data [i];
				});
			}
		}

		public void Add_(float value)
		{
			if (dataOnGpu) {
				AddScalarGPU_ (value);
				return;
			} else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						data [i] += value;
				});
			}
		}

        public void Floor_()
        {
            if (dataOnGpu)
            {
                FloorGPU_();
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    Data[i] = (float)(Math.Floor(Data[i]));
            });
        }

		public void Mul_(FloatTensor x)
		{
			SameSizeDimensionsShapeAndLocation(ref x);

			if (dataOnGpu) {
				MulElemGPU_ (x);

			} else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						data [i] *= x.data [i];
				});
			}
		}

		public void Mul_(float value)
		{
			if (dataOnGpu) {
				MulScalarGPU_ (value);
				return;
			} else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						data [i] *= value;
				});
			}
		}

        public void Sigmoid_()
        {
            if (dataOnGpu)
            {
                SigmoidGPU_();
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    double s = Math.Exp((double)data[i]);
                    data[i] = (float)(s / (1.0f + s));
                }
            });
        }

		public void Sub_(FloatTensor x)
		{
			SameSizeDimensionsShapeAndLocation(ref x);

			if (dataOnGpu) {
				SubElemGPU_ (x);

			} else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						data [i] -= x.data [i];
				});
			}
		}

		public void Sub_(float value)
		{
			if (dataOnGpu) {
				SubScalarGPU_ (value);
				return;
			} else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++)
						data [i] -= value;
				});
			}
		}


        public void Zero_()
        {
            if (dataOnGpu)
            {
                ZeroGPU_();
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = data.Length * (workerId + 1) / nCpu;
                for (int i = data.Length * workerId / nCpu; i < max; i++)
                    data[i] = 0;
            });
        }
    }
}