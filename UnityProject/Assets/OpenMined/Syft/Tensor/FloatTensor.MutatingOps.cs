using System;
using System.Threading.Tasks;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor

    {

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

    }
}
