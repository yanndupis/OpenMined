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

        public void Add_(float value)
        {
            if (dataOnGpu)
            {
				AddScalarGPU_(value);
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                    data[i] += value;
            });
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