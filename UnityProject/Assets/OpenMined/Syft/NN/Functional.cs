using System;
using System.Collections.Generic;
using System.Linq;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.NN
{
    public static class Functional
    {
        // TODO: Softmax will run on GPU, when below OPS have a GPU implementation!
        // TODO: Improve the implementation!!!
        public static FloatTensor Softmax(FloatTensor input, int dim = -1)
        {
   
            //TODO: GPU support
            var gpu = false;
            if (input.DataOnGpu)
            {   
                input.Cpu();
                gpu = true;
            }
            
            var _dim = (dim == -1) ? input.Shape.Length - 1 : dim;

            var outerSize = 1;
            var innerSize = 1;
            var dimSize = input.Shape[_dim];
            
            for (var i = 0; i < _dim; ++i)
                outerSize *= input.Shape[i];
            
            for (var i = _dim + 1; i < input.Shape.Length; ++i)
                innerSize *= input.Shape[i];
            
            var dimStride = innerSize;
            var outerStride = dimSize * dimStride;
            
            var result = input.Exp();

            for (var i = 0; i < outerSize * innerSize; i++)
            {
                int outerIdx = i / innerSize;
                int innerIdx = i % innerSize;
                
                // works for contiguous!!
                var inputData = outerIdx * outerStride + innerIdx;
                
                float sum = 0;
                for (var d = 0; d < dimSize; d++)
                    sum += result.Data[inputData + d * dimStride];
                
                for (var d = 0; d < dimSize; d++)
                    result.Data[inputData + d * dimStride] = result.Data[inputData + d * dimStride] / sum;
            }


            if (gpu)
            {
                result.Gpu(input.Shader);
            }

            return result;
        }
    }
}