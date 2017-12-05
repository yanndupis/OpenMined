using System;
using System.Threading.Tasks;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
public partial class FloatTensor
{


public void Pow_ (FloatTensor x)
{
	SameSizeDimensionsShapeAndLocation (ref x);

	if (dataOnGpu) {
		PowElemGPU_ (x);
	} else {
		var nCpu = SystemInfo.processorCount;
		Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++) {
					        data [i] = (float)Math.Pow ((double)data [i], x.data [i]);
					}
				});
	}
}

public void Pow_ (float value)
{
	if (dataOnGpu) {
		PowScalarGPU_ (value);
		return;
	}
	var nCpu = SystemInfo.processorCount;
	Parallel.For (0, nCpu, workerId => {
				var max = size * (workerId + 1) / nCpu;
				for (var i = size * workerId / nCpu; i < max; i++) {
				        data [i] = (float)Math.Pow ((double)data [i], value);
				}
			});
}

}
}
