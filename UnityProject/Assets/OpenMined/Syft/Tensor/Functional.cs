using System;
using System.Collections.Generic;
using OpenMined.Syft.Tensor.Factories;

namespace OpenMined.Syft.Tensor
{
    public class Functional
    {
        /*public static FloatTensor Concatenate(FloatTensorFactory factory, List<int> tensor_ids, int axis, FloatTensor result = null)
        {
            if (axis == 0)
            {
                List<FloatTensor> tensors = new List<FloatTensor>();
                foreach (int id in tensor_ids)
                {
                    tensors.Add(factory.Get(id));
                }

                FloatTensor first = tensors[0];

                if(first.DataOnGpu)
                    throw new NotImplementedException("Can't concatenate GPU tensors yet");
                
                int num_new_rows = 0;
                
                foreach (FloatTensor tensor in tensors)
                {
                    if (tensor.Shape.Length != first.Shape.Length)
                    {
                        throw new InvalidOperationException("Tensors do not have the same number of dimensions.");
                    }

                    for (int i = 0; i < tensor.Shape.Length; i++)
                    {
                        if (i != axis)
                        {
                            if (tensor.Shape[i] != first.Shape[i])
                            {
                                throw new InvalidOperationException("Tensors do not have the same shape.");
                            }
                        }
                    }

                    if (tensor.DataOnGpu != first.DataOnGpu)
                    {
                        throw new InvalidOperationException("All tensors must be on the same device...");
                    }

                    num_new_rows += tensor.Shape[axis];
                }

                int[] concat_shape = new int[first.Shape.Length];

                for (int i = 0; i < first.Shape.Length; i++)
                {
                    if (i == axis)
                    {
                        concat_shape[i] = num_new_rows;
                    }
                    else
                    {
                        concat_shape[i] = first.Shape[i];
                    }
                }

                first.HookGraph(result,)
                                

            }
            else
            {
                throw new NotImplementedException();
            }
        }*/
    }
}