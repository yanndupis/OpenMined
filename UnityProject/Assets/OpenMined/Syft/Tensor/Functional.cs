﻿using System;
using System.Collections.Generic;
using OpenMined.Syft.Tensor.Factories;

namespace OpenMined.Syft.Tensor
{
    public class Functional
    {
        // opposite of concatenate
        public static List<int> Batchify(FloatTensor input,int dim, int batch_size)
        {
            List<int> batches = new List<int>();
            
            List<int> indices = new List<int>();
            for (int i = 0; i < input.Shape[dim]; i++)
            {
                indices.Add(i);
            }

            for (int i = 0; i < input.Shape[dim] / batch_size; i++)
            {
                batches.Add(input.IndexSelect(indices.GetRange(i*batch_size,batch_size),dim).Id);
            }
            
            if(input.Shape[dim] % batch_size != 0)
                batches.Add(input.IndexSelect(indices.GetRange(input.Shape[dim] - (input.Shape[dim] % batch_size),input.Shape[dim] % batch_size),dim).Id);

            return batches;
        }
        
        // opposite of batchify
        public static FloatTensor Concatenate(FloatTensorFactory factory, List<int> tensor_ids, int axis, FloatTensor result = null)
        {
            FloatTensor[] tensors = new FloatTensor[tensor_ids.Count-1];
            for(int i=0; i<tensor_ids.Count-1; i++)
            {
                tensors[i] = factory.Get(tensor_ids[i+1]);
            }

            FloatTensor first = factory.Get(tensor_ids[0]);

            if(first.DataOnGpu)
                throw new NotImplementedException("Can't concatenate GPU tensors yet");

            int num_new_rows = 0;

            List<IntTensor> int_tensors_for_index_add = new List<IntTensor>();

            int[] first_indices = new int[first.Shape[axis]];
            for (int i = 0; i < first.Shape[axis]; i++) first_indices[i] = i + num_new_rows;
            int_tensors_for_index_add.Add(factory.ctrl.intTensorFactory.Create(_shape: new int[1] {first.Shape[axis]},_data:first_indices));
            
            num_new_rows += first.Shape[axis];
            
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

                int[] indices = new int[tensor.Shape[axis]];
                for (int i = 0; i < tensor.Shape[axis]; i++) indices[i] = i + num_new_rows;
                int_tensors_for_index_add.Add(factory.ctrl.intTensorFactory.Create(_shape: new int[1] {tensor.Shape[axis]},_data:indices));
                
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
            
            result = first.HookGraph(ref result, tensor_inputs: tensors, creation_op: "concatenate_"+axis, inline: false, resultShape:concat_shape, indices:int_tensors_for_index_add.ToArray());
            
            if (axis != 0)
            {
                result.IndexAdd(int_tensors_for_index_add[0], axis, first, inline: true);

                for (int i = 0; i < tensors.Length; i++)
                {
                    result.IndexAdd(int_tensors_for_index_add[i+1],axis,tensors[i],inline:true);
                }
            } 
            else 
            {
                
                int result_i = 0;

                for (int i = 0; i < first.Data.Length; i++)
                {
                    result.Data[result_i] = first.Data[i];
                    result_i += 1;
                }
                
                foreach (FloatTensor tensor in tensors)
                {
                    for (int i = 0; i < tensor.Data.Length; i++)
                    {
                        result.Data[result_i] = tensor.Data[i];
                        result_i += 1;
                    }
                }

            }
            return result;            
        }
        public static FloatTensor Ones(FloatTensorFactory factory, int[] dims)
        {
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(dims);
            result.Add(1.0F, inline: true);
            return result;
        }
        public static FloatTensor Randn(FloatTensorFactory factory, int[] dims)
        {
        int dims_prod = 1;
            foreach (int dim in dims)
            {
                dims_prod *= dim;
            }
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(dims);
            for (int i = 0; i < dims_prod; i++)
            {
                // Reference: https://stackoverflow.com/questions/218060/random-gaussian-variables
                float u1 = 1.0F - UnityEngine.Random.value;
                float u2 = 1.0F - UnityEngine.Random.value;
                result.Data[i] = Convert.ToSingle(Math.Sqrt(-2.0F * Math.Log(u1)) * Math.Sin(2.0F * Math.PI * u2));
            }
            return result.View(dims);
        }
        public static FloatTensor Random(FloatTensorFactory factory, int[] dims)
        {
            int dims_prod = 1;
            foreach (int dim in dims)
            {
                dims_prod *= dim;
            }
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(dims);
            for (int i = 0; i < dims_prod; i++)
            {
                result.Data[i] = UnityEngine.Random.value;
            }
            return result.View(dims);
        }
        public static FloatTensor Zeros(FloatTensorFactory factory, int[] dims)
        {
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(dims);
            return result;
        }
    }
}