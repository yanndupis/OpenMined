using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        internal FloatTensor emptyTensorCopy(bool hook_graph = false, FloatTensor result = null)
        {

            if (hook_graph)
            {
                result = HookGraph(ref result, "emptyTensorCopy_Hooked", false);
                result.Zero_();
                return result;
            }
            else
            {
                
                result = factory.Create(
                    _shape: this.shape,
                    _data: data,
                    _dataBuffer: dataBuffer,
                    _shapeBuffer: shapeBuffer,
                    _shader: shader,
                    _copyData: true,
                    _dataOnGpu: dataOnGpu,
                    _autograd: autograd,
                    _keepgrads: keepgrads,
                    _creation_op: "emptyTensorCopy");
            
                result.Zero_();

                return result;
            }
            
        }
        
        // parameters are overrides
        public FloatTensor Copy(FloatTensor result = null)
        {
            result = HookGraph(ref result, "copy", false);
            result.Zero_();
            result.Add(this, inline: true);
            
            return result;
        }

        public FloatTensor createZerosTensorLike() {
            FloatTensor new_tensor = this.emptyTensorCopy ();
            new_tensor.Zero_ ();
            return new_tensor;
        }

        public FloatTensor createOnesTensorLike() {
            FloatTensor new_tensor = this.emptyTensorCopy();
            new_tensor.Zero_ ();
            new_tensor.Add ((float)1,true);
            return new_tensor;
        }
        
		public FloatTensor Abs(bool inline = false, FloatTensor result = null)
		// Returns a new Tensor with the smallest integer greater than or equal to each element
		{
            result = HookGraph(ref result, "abs", inline);		

			if (dataOnGpu) {
				if (inline) { AbsGPU_ (); return this; }
				else { return AbsGPU (result); }
			}
			else {
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++) {
					        result.Data [i] = (float)(Math.Abs (Data [i]));
					}
				});
			}
			return result;
		}
        
		public FloatTensor Add(FloatTensor x, bool inline = false, FloatTensor result = null, bool override_checks= false)
		{
		    
		    if ((!IsContiguous() || !x.IsContiguous()) && override_checks == false) 
		        throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");

		    if (!override_checks)
		    {
		        // Check if both tensors are compatible for sub - fallback to scalar version if either tensor's size == 1
		        if (SameSizeDimensionsShapeAndLocation(ref x))
		        {
		            if (x.Size == 1)
		            {
		                return this.Add(x.Expand(shape).Contiguous(), inline);
		            }
		            else if (this.Size == 1)
		            {
		                if (inline)
		                {
		                    throw new InvalidOperationException("Tensor sizes don't match");
		                }

		                return x.Add(this.Expand(x.shape).Contiguous());
		            }
		            else
		            {
		                throw new InvalidOperationException();
		            }
		        }
		    }

		    result = HookGraph (ref result, ref x, "add_elem", inline);


		    if (dataOnGpu)
		    {
		        if (inline)
		        {
		            if (autograd)
		                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");


		            AddElemGPU_(x);
		            return this;
		        }
		        else
		        {
		            return AddElemGPU(x, result);
		        }
		    }

		    var nCpu = SystemInfo.processorCount;
            Parallel.For (0, nCpu, workerId => {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++) {
                        result.Data [i] = x.Data [i] + Data [i];
                }
            });


			return result;
		}

        
        public FloatTensor Add(float value, bool inline = false, FloatTensor result = null)
        {
            result = HookGraph (ref result, value, "add_scalar", inline);

            if (dataOnGpu) {
                result.Gpu (shader);
                if (inline) { AddScalarGPU_ (value); return this; }
                else { return AddScalarGPU (value, result); }
            }
            else {
                var nCpu = SystemInfo.processorCount;
                Parallel.For (0, nCpu, workerId => {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++) {
                        result.Data [i] = value + Data [i];
                    }
                });
            }
            return result;
        }

		public FloatTensor Acos (bool inline = false)
		{
			if (dataOnGpu) {
				if (inline) { AcosGPU_(); return this;}
				else { return AcosGPU (); }
			} else {
				FloatTensor result = inline ? this : this.emptyTensorCopy();
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++) {
					        var d = (double)Data [i];
					        result.Data [i] = (float)System.Math.Acos (d);
					}
				});
				return result;
			}
		}
        
        public FloatTensor AddMatrixMultiply(FloatTensor tensor1, FloatTensor tensor2)
        {
            if (!IsContiguous() || !tensor1.IsContiguous() || !tensor2.IsContiguous()) {
                throw new InvalidOperationException("All tensors must be contiguous, call Contiguous() to convert");
            }

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

        public FloatTensor AddMatrixVectorProduct(FloatTensor matrix, FloatTensor vector)
        {
            if (!IsContiguous() || !matrix.IsContiguous() || !vector.IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }
            
            var gpu = dataOnGpu & matrix.DataOnGpu & vector.DataOnGpu;
            var cpu = !(dataOnGpu | matrix.DataOnGpu | vector.DataOnGpu);

            var ref_shape = this.Shape;
            var matrix_shape = matrix.Shape;
            var vector_shape = vector.Shape;

            if (ref_shape.Length != 1)
                throw new InvalidOperationException(
                    "Cannot perform this operation on a tensor with more than one dimension");
            if (ref_shape[0] != vector_shape[0])
                throw new InvalidOperationException(String.Format(
                    "Cannot add matrix-vector product to tensor: {0} & {1}.", ref_shape[0], vector_shape[0]));
            if (matrix_shape[1] != vector_shape[0])
                throw new InvalidOperationException(String.Format("Last dimension of matrix doesn't match: {0} vs {1}.",
                    matrix_shape[1], vector_shape[0]));

            if (gpu)
            {
                AddMatrixVectorProductGPU(matrix, vector);
            }
            else if (cpu)
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId =>
                {
                    var max = size * (workerId + 1) / nCpu;
                    for (var idx = size * workerId / nCpu; idx < max; idx++)
                    {
                        for (var j = 0; j < ref_shape[0]; j++)
                        {
                            this[idx] += vector[j] * matrix[j + (idx * ref_shape[0])];
                        }
                    }
                });
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }

            return this;
        }
     
		public FloatTensor Asin ( bool inline = false)
		{
			if (dataOnGpu) {
				if (inline) { AsinGPU_(); return this;}
				else { return AsinGPU (); }
			} else {
				var result = inline ? this : this.emptyTensorCopy();
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++) {
					        var d = (double)Data [i];
					        result.Data [i] = (float)System.Math.Asin (d);
					}
				});

				return result;
			}
		}

		public FloatTensor Atan (bool inline = false)
		{
			if (dataOnGpu) {
				if (inline) { AtanGPU_(); return this;}
				else { return AtanGPU (); }
			} else {
				var result = inline ? this : this.emptyTensorCopy();
				var nCpu = SystemInfo.processorCount;
				Parallel.For (0, nCpu, workerId => {
					var max = size * (workerId + 1) / nCpu;
					for (var i = size * workerId / nCpu; i < max; i++) {
					        var d = (double)Data [i];
					        result.Data [i] = (float)System.Math.Atan (d);
					}
				});

				return result;
			}
		}

		public FloatTensor Ceil(bool inline = false)
        {
            // Returns a new Tensor with the smallest integer greater than or equal to each element
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                //TODO: Fix GPU operations. https://github.com/OpenMined/OpenMined/issues/126
                result.Gpu(shader);
                if (!inline) return CeilGPU(result);
                CeilGPU_();
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Ceiling(x)).ToArray();
            return result;
        }

        public FloatTensor Contiguous(FloatTensor result = null)
        {

            if (DataOnGpu)
                throw new NotSupportedException();
         
            result = HookGraph(ref result, "contiguous", false, shape);

            int[] dim_indices = new int[strides.Length];
            
            for (int i = 0; i < result.Data.Length; i++)
            {    
                result.DataIndex2DimIndices(i, ref dim_indices);
                result.data[i] = this.data[this.DimIndices2DataIndex(ref dim_indices)];
            }   
            
            return result;
        }
        
        public FloatTensor Cos(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return CosGPU();
                CosGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Cos((double) x)).ToArray();
            return result;
        }

        public FloatTensor Cosh(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return CoshGPU();
                CoshGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Cosh((double) x)).ToArray();
            return result;
        }

        public FloatTensor CumProd(int dim, bool inline = false , FloatTensor result = null)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            
            result = HookGraph(ref result, "cumprod_"+dim, inline);
            result.Zero_();
            result.Add(this, inline: true);

            int[] temp_shape = new int[] {1, shape[dim], 1};

        
            for (int i = 0; i < dim; i++)
            {
                temp_shape[0] *= shape[i];
            }
        
            for (int i = dim+1; i < shape.Length; i++)
            {
                temp_shape[2] *= shape[i];
            }
            
            var result_3d = result.View(temp_shape);

            int[] temp_index = new int[] {0, 0, 0};
            float cumprod = 1;
            
            for (int i = 0; i < result_3d.shape[0]; i++)
            {
                temp_index[0] = i;

                for (int j = 0; j < result_3d.shape[2]; j++)
                {
                    temp_index[2] = j;
                    
                    cumprod = 1;
                    for (var k = 0; k < result_3d.Shape[1]; k++)
                    {
                        temp_index[1] = k;
                        int result_data_index = result_3d.DimIndices2DataIndex(ref temp_index);

                        cumprod *= result_3d.Data[result_data_index];
                        result_3d.Data[result_data_index] = cumprod;
                    }
                }
            }

            return result_3d.View(shape);
        }      
        
        public FloatTensor CumSum(int dim, bool inline = false , FloatTensor result = null)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            
            result = HookGraph(ref result, "cumsum_"+dim, inline);
            result.Zero_();
            result.Add(this, inline: true);

            int[] temp_shape = new int[] {1, shape[dim], 1};

        
            for (int i = 0; i < dim; i++)
            {
                temp_shape[0] *= shape[i];
            }
        
            for (int i = dim+1; i < shape.Length; i++)
            {
                temp_shape[2] *= shape[i];
            }
            
            var result_3d = result.View(temp_shape);

            int[] temp_index = new int[] {0, 0, 0};
            float cumsum = 0;
            
            for (int i = 0; i < result_3d.shape[0]; i++)
            {
                temp_index[0] = i;

                for (int j = 0; j < result_3d.shape[2]; j++)
                {
                    temp_index[2] = j;
                    
                    cumsum = 0;
                    for (var k = 0; k < result_3d.Shape[1]; k++)
                    {
                        temp_index[1] = k;
                        int result_data_index = result_3d.DimIndices2DataIndex(ref temp_index);

                        cumsum += result_3d.Data[result_data_index];
                        result_3d.Data[result_data_index] = cumsum;
                    }
                }
            }

            return result_3d.View(shape);
        }

        public FloatTensor Div(FloatTensor x, bool inline = false, FloatTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            // Check if both tensors are compatible for sub - fallback to scalar version if either tensor's size == 1
            if (SameSizeDimensionsShapeAndLocation(ref x))
            {
                if (x.Size == 1)
                {
                    return this.Div(x.Expand(shape).Contiguous(), inline);
                }
                else if (this.Size == 1)
                {
                    if (inline)
                    {
                        throw new InvalidOperationException("Tensor sizes don't match");
                    }

                    return x.Div(this.Expand(x.shape).Contiguous()).Pow(-1);
                }
                else
                {
                    throw new InvalidOperationException();
                }
            }
            result = HookGraph(ref result, ref x, "div_elem", inline);
            
            if (dataOnGpu & x.dataOnGpu)
            {
                result.Gpu(shader);
                if (inline)
                {
                    if (autograd)
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    DivElemGPU_(x);
                    return this;
                }
                result = DivElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a / b).ToArray();
            }

            return result;
        }

        public FloatTensor Div(float value, bool inline = false, FloatTensor result = null)
        {
            result = HookGraph (ref result, value, "div_scalar", inline);
            
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return DivScalarGPU(value, result);
                DivScalarGPU_(value);
                return this;
            }
            result.Data = data.AsParallel().Select(x => x / value).ToArray();
            return result;
        }

        public FloatTensor Exp(bool inline = false)
        {
            //var result = new FloatTensor(_ctrl:ctrl, _shape:shape, _shader:this.shader);
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return ExpGPU();
                ExpGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Exp((double) x)).ToArray();
            return result;
        }
        
        public FloatTensor Expand(int[] sizes) {
			if (sizes.Length == Shape.Length) {
				return ExpandFixedDimensions(sizes);
			} else if (sizes.Length > Shape.Length) {
				return expandNewDimensions(sizes);
			} else {
			    throw new InvalidOperationException(String.Format("Number of sizes provided must be greater than or equal to the number of dimensions in tensor"));
			}
		}

        private FloatTensor ExpandFixedDimensions(int[] sizes, FloatTensor result = null)
        {

            // TODO: make more complicated version which does not copy data
            result = HookGraph(ref result, "expand", false, shape);
            result.Add(this, inline: true,override_checks:true);
		    
            for (int i = 0; i < shape.Length; i++) {
                if (sizes[i] != -1 && sizes[i] != shape[i]) {
                    if (shape[i] == 1 || strides[i] == 0) {
                        result.strides[i] = 0;
                        result.shape[i] = sizes[i];
                    } else {
                        throw new InvalidOperationException (String.Format ("Cannot expand dimension {0}, not a singleton ({1})", i, shape[i]));
                    }
                }
            }

            return result;
        }

        private FloatTensor expandNewDimensions(int[] sizes) {
            FloatTensor result = factory.Create(_data: data, _shape: shape, _shader: shader, _copyData: false);

            int diffLength = sizes.Length - shape.Length;
			
            // sets new strides to zero on initialization
            int[] newStrides = new int[sizes.Length];
            int[] newShape = new int[sizes.Length];

            for (int i = 0; i < diffLength; i++) {
                // sets new shape
                if (sizes[i] != -1) {
                    newShape[i] = sizes[i];
                } else {
                    throw new InvalidOperationException (String.Format ("Cannot set new dimension {0} to -1", i));
                }
            }
			
            for (int i = diffLength; i < sizes.Length; i++) {
                var oldIndex = i - diffLength;
				
                // fill in old strides/shape
                newStrides[i] = strides[oldIndex];
                newShape[i] = shape[oldIndex];
				
                // modify any old strides/shapes
                if (sizes[i] != -1 && sizes[i] != shape[oldIndex]) {
                    if (shape[oldIndex] == 1 || strides[oldIndex] == 0) {
                        newStrides[i] = 0;
                        newShape[i] = sizes[i];
                    } else {
                        throw new InvalidOperationException (String.Format ("Cannot expand dimension {0}, not a singleton ({1})", i, shape[i]));
                    }
                }
            }

            result.shape = newShape;
            result.strides = newStrides;
			
            return result;
        }        

        internal void ForEach(int dim, Action<float[], int, int> iterator)
        
        {
            int interations = size / shape[dim];
            int values = shape[dim];
            var stride = strides[dim];
            MultiThread.For(interations, (i, len) =>
            {
                var temp = new float[values];
                var offset = GetDimReduceOffset(i, values, stride);

                for (int v = 0; v < values; v++)
                {
                    temp[v] = this[offset + v * stride];
                }

                iterator(temp, offset, stride);
            });
        }        
        
        public FloatTensor Floor(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return FloorGPU(result);
                FloorGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Floor(x)).ToArray();
            return result;
        }
        
        public FloatTensor IndexSelect(IntTensor indices, int dim, FloatTensor result = null)
        {
            if (DataOnGpu)
            {
                throw new NotImplementedException();
            }

            if (indices.Shape.Length != 1)
            {
                throw new NotImplementedException("Indices must be a list");
            }

            
            int[] temp_shape = new int[] {1, shape[dim], 1};

            for (int i = 0; i < dim; i++)
            {
                temp_shape[0] *= shape[i];
            }
    
            for (int i = dim+1; i < shape.Length; i++)
            {
                temp_shape[2] *= shape[i];
            }
                
            var self_3d = this.View(temp_shape);

            int[] result_3d_shape = new int[] {temp_shape[0], indices.Shape[0], temp_shape[2]};

            result = HookGraph(ref result, "index_select_" + dim + "_" + indices.Id, false, result_3d_shape, indices:indices);
            
            int[] temp_index = new int[] {0, 0, 0};
        
            for (int i = 0; i < self_3d.shape[0]; i++)
            {
                temp_index[0] = i;

                for (int j = 0; j < self_3d.shape[2]; j++)
                {
                    temp_index[2] = j;
                
                    for (var k = 0; k < indices.Shape[0]; k++)
                    {
                        temp_index[1] = indices.Data[k];
                        int result_data_index = self_3d.DimIndices2DataIndex(ref temp_index);

                        temp_index[1] = k;
                        result.Data[result.DimIndices2DataIndex(ref temp_index)] = self_3d.Data[result_data_index];
                        
                    }
                        
                }
                    
            }

            int[] result_dim = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                if (i != dim)
                {
                    result_dim[i] = shape[i];
                }
                else
                {
                    result_dim[i] = indices.Shape[0];
                }
            }
            
            return result.View(result_dim);
        }

        public FloatTensor IndexAdd(IntTensor indices, int dim, FloatTensor x, FloatTensor result = null)
        {
            if (DataOnGpu)
            {
                throw new NotImplementedException();
            }
            
            if (indices.Shape.Length != 1)
            {
                throw new NotImplementedException("Indices must be a list");
            }

            /*if (indices.Shape[dim] != x.Shape[dim])
            {
                throw new IndexOutOfRangeException("Indices and Input Sum must have same number of rows");
            }*/
            
            int[] temp_shape = new int[] {1, shape[dim], 1};

            for (int i = 0; i < dim; i++)
            {
                temp_shape[0] *= shape[i];
            }
    
            for (int i = dim+1; i < shape.Length; i++)
            {
                temp_shape[2] *= shape[i];
            }
                
            var self_3d = this.View(temp_shape);
            var x_3d = x.View(new int[] {temp_shape[0], indices.Shape[0], temp_shape[2]});

            // TODO: Hook Autograd should support this
            result = HookGraph(ref result, "index_add_dim:" + dim + "_" + indices.Id + "_" + x.Id, false, temp_shape);
            result.Zero_();
            result.Add(this, inline: true, override_checks: true);
            
            int[] temp_index = new int[] {0, 0, 0};
            
            for (int i = 0; i < self_3d.shape[0]; i++)
            {
                temp_index[0] = i;

                for (int j = 0; j < self_3d.shape[2]; j++)
                {
                    temp_index[2] = j;
                
                    for (var k = 0; k < indices.Shape[0]; k++)
                    {
                        temp_index[1] = k;
                        int x_dataindex = x_3d.DimIndices2DataIndex(ref temp_index);
                        
                        temp_index[1] = indices.Data[k];
                        result.Data[result.DimIndices2DataIndex(ref temp_index)] += x_3d.Data[x_dataindex];

                    }
                        
                }
                    
            }

            return result.View(shape);
        }

        public bool IsContiguous()
        {
            long z = 1;
            int d;
            for(d = shape.Length-1; d >= 0; d--)
            {
                if(shape[d] != 1)
                {
                    if (strides[d] == z) {
                        z *= shape[d];
                    } else {
                        return false;
                    }
                }
            }
            return true;
        }

        public FloatTensor Log1p(bool inline = false)
        {	
        	var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
            	if (!inline) return Log1pGPU();
            	Log1pGPU_();
            	return this;
            }
            result.Data = data.AsParallel().Select(x => (float) (Math.Log(1 + x))).ToArray();
            return result;
        }
        
        public FloatTensor Log(bool inline = false, FloatTensor result = null)
        {	
            result = HookGraph(ref result, "log", inline);

            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            result.Data = data.AsParallel().Select(x => (float) (Math.Log(x))).ToArray();
            
            return result;
        }

        public FloatTensor Max(int dim = -1, bool keepdim = false)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc > val ? acc : val, (val, len) => val, creation_op:"max_"+dim);
        }        

        public FloatTensor Mean(int dim = -1, bool keepdim = false)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }
            
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc + val, (val, len) => val / (float) len, creation_op:"mean_"+dim);
        }        
        
        public FloatTensor Min(int dim = -1, bool keepdim = false)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc < val ? acc : val, (val, len) => val, creation_op:"min_"+dim);
        }

        
        public FloatTensor MM(FloatTensor x, FloatTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");
            }

            if (this.shape.Length != 2 || x.shape.Length != 2)
            {
                throw new InvalidOperationException(
                    "Cannot do MM on tensors that aren't 2 dimentional. Try calling view() to reshape");
            }
            
            result = HookGraph(ref result, ref x,  "mm", false, new int[]{shape[0],x.shape[1]});
            
            result.AddMatrixMultiply(this, x);

            return result;
        }

        public FloatTensor Mul(FloatTensor x, bool inline = false, FloatTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");
            }

            // Check if both tensors are compatible for sub - fallback to scalar version if either tensor's size == 1
            if (SameSizeDimensionsShapeAndLocation(ref x))
            {
                if (x.Size == 1)
                {
                    return this.Mul(x.Expand(shape).Contiguous(), inline);
                }
                else if (this.Size == 1)
                {
                    if (inline)
                    {
                        throw new InvalidOperationException("Tensor sizes don't match");
                    }

                    return x.Mul(this.Expand(x.shape).Contiguous());
                }
                else
                {
                    throw new InvalidOperationException();
                }
            }

            result = HookGraph(ref result, ref x, "mul_elem", inline);

            if (dataOnGpu && x.dataOnGpu)
            {
                if (inline)
                {
                    if (autograd)
                    {
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    }
                    MulElemGPU_(x);
                    return this;
                }
                result = MulElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a * b).ToArray();
            }

            return result;
        }

        public FloatTensor Mul(float value, bool inline = false, FloatTensor result = null)
        {
            result = HookGraph (ref result, value, "mul_scalar", inline);

            if (dataOnGpu)
            {
                if (!inline) return MulScalarGPU(value, result);
                MulScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => x * value).ToArray();
            return result;
        }

        public FloatTensor Neg(bool inline = false, FloatTensor result = null)
        {
            result = HookGraph(ref result, "neg", inline);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return NegateGPU();
                NegateGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => -x).ToArray();
            return result;
        }
        
        public FloatTensor Pow(FloatTensor x, bool inline = false, FloatTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");
            }

            // Check if both tensors are compatible for sum
            SameSizeDimensionsShapeAndLocation(ref x);

            result = HookGraph(ref result, ref x, "pow_elem", inline);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return PowElemGPU(x, result);
                result.PowElemGPU_(x);
                return this;
            }

            result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => (float) Math.Pow((double) a, b))
                .ToArray();
            
            return result;
        }

        public FloatTensor Pow(float value, bool inline = false, FloatTensor result = null)
        {
            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");
            
            result = HookGraph(ref result, value, "pow_scalar", inline);
            
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return PowScalarGPU(value, result);
                PowScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Pow((double) x, value)).ToArray();
            
            return result;
        }

        public FloatTensor Prod(int dim = -1, bool keepdim = false)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }
            
            // TODO: Implement GPU op. with GPU tests.
            return Reduce(dim, keepdim, (acc, val, index, arr) => acc * val, (val, len) => val, creation_op:"prod_"+dim);
        }        
        
        public FloatTensor Reciprocal(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return ReciprocalGPU();
                ReciprocalGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (float) 1/x).ToArray();
            return result;
        }
   
        /*** Reduce Functions ***/
 public FloatTensor Reduce(
            Func<float, float, int, float[], float> reducer,
            Func<float, int, float> mapper, string creation_op, FloatTensor result = null
        )
        {
            int[] outDims = {1};
            var output = new float[1];
            
            result = HookGraph(ref result, creation_op, false, outDims);
            
            result.data[0] = mapper(MultiThread.Reduce(data, reducer), Size);

            return result;
        }

        public FloatTensor Reduce(
            int dim,
            bool keepdim,
            Func<float, float, int, float[], float> reducer,
            Func<float, int, float> mapper,
            string creation_op = null,
            FloatTensor result = null
        )
        {
            int len = shape.Length;

            if (dim < 0)
            {
                return Reduce(reducer, mapper, creation_op:creation_op);
            }

            AssertDim(dim, len);

            if (len == 1)
            {
                keepdim = true;
            }

            var stride = strides[dim];
            int values = shape[dim];

            int outSize = 1;
            var outDims = keepdim ? new int[len] : new int[len - 1];

            for (int i = 0; i < len; i++)
            {
                if (i < dim)
                {
                    outDims[i] = shape[i];
                }
                else if (i > dim)
                {
                    outDims[keepdim ? i : i - 1] = shape[i];
                }
                else if (i == dim)
                {
                    if (keepdim)
                    {
                        outDims[i] = 1;
                    }

                    continue;
                }

                outSize *= shape[i];
            }
            
            result = HookGraph(ref result, creation_op, false, outDims);

            _dimForEach(outSize, values, stride, (vals, index, length) =>
            {
                var acc = vals[0];

                for (int i = 1; i < length; i++)
                {
                    acc = reducer(acc, vals[i], i, vals);
                }

                result.data[index] = mapper(acc, length);
            });

            return result;
        }        
        
        public FloatTensor ReLU(bool inline = false, FloatTensor result = null)
        {

            result = HookGraph(ref result, "relu", inline);

            if (dataOnGpu)
            {
                
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Max((double) x,0)).ToArray();
            return result;
        }
      
        public FloatTensor Remainder(float divisor, bool inline = false)
        {
            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");
            if (autograd)
                throw new InvalidOperationException("Autograd not available for Remainder.");

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { RemainderScalarGPU_(divisor); return this; }
                else { result = RemainderScalarGPU(result, divisor); }
            }
            else
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId => {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                    {
                        result[i] = this[i] % divisor;
                    };
                });
            }

            return result;
        }

        public FloatTensor Remainder(FloatTensor divisor, bool inline = false)
        {
            if (!IsContiguous() || !divisor.IsContiguous()) {
                throw new InvalidOperationException ("All tensor must be contiguous, call Contiguous() to convert");
            }

            SameSizeDimensionsShapeAndLocation(ref divisor);
            if (inline & autograd)
                throw new InvalidOperationException("Cannot call inline functions if you intend to run backprop.");
            if (autograd)
                throw new InvalidOperationException("Autograd not available for Remainder.");

            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline)
                {
                    RemainderElemGPU_(divisor);
                    return this;
                }
                else
                {
                    result = RemainderElemGPU(divisor,result);
                }
            }
            else
            {
                var nCpu = SystemInfo.processorCount;
                Parallel.For(0, nCpu, workerId => {
                    var max = size * (workerId + 1) / nCpu;
                    for (var i = size * workerId / nCpu; i < max; i++)
                    {
                        result[i] = this[i] % divisor[i];
                    };
                });
            }

            return result;
        }   
        
        public FloatTensor Round(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return RoundGPU();
                RoundGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (float) Math.Round(x)).ToArray();
            return result;
        }

        public FloatTensor Rsqrt(bool inline = false)
        {   
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return RsqrtGPU();
                RsqrtGPU_();
                return this;
            }
            result.Data = data.AsParallel().Select(x => 1 / (float) Math.Sqrt(x)).ToArray();
            return result;
        }

        public IntTensor Sample(int dim, IntTensor result = null)
        {
            
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            if (dim == -1)
            {
                
                result = factory.ctrl.intTensorFactory.Create(shape);
                
                for (int i = 0; i < size; i++)
                {
                    if (UnityEngine.Random.value < data[i])
                    {
                        result.Data[i] = 1;
                    }
                    else
                    {
                        result.Data[i] = 0;
                    }

                }

                return result;
            }
            else
            { 

                int[] temp_shape = new int[] {1, shape[dim], 1};

    
                for (int i = 0; i < dim; i++)
                {
                    temp_shape[0] *= shape[i];
                }
    
                for (int i = dim+1; i < shape.Length; i++)
                {
                    temp_shape[2] *= shape[i];
                }
        
                result = factory.ctrl.intTensorFactory.Create(new int[] {temp_shape[0], 1, temp_shape[2]});
                
                var result_3d = this.View(temp_shape);

                int[] temp_index = new int[] {0, 0, 0};
                float cumsum = 0;
                float random = 0;
        
                for (int i = 0; i < result_3d.shape[0]; i++)
                {
                    temp_index[0] = i;

                    for (int j = 0; j < result_3d.shape[2]; j++)
                    {
                        temp_index[2] = j;
                
                        cumsum = 0;
                        random = UnityEngine.Random.value;
                        for (var k = 0; k < result_3d.Shape[1]; k++)
                        {
                            temp_index[1] = k;
                            int result_data_index = result_3d.DimIndices2DataIndex(ref temp_index);

                            cumsum += result_3d.Data[result_data_index];
                            if (random > (1 - cumsum))
                            {
                                temp_index[1] = 0;
                                result[result.DimIndices2DataIndex(ref temp_index)] = k;
                                break;
                            }
                        }
                        
                    }
                    
                }

                int[] final_shape = new int[shape.Length - 1];
                int h = 0;
                for (int i = 0; i < shape.Length; i++)
                {
                    if (i != dim)
                    {
                        final_shape[h] = shape[i];
                        h += 1;
                    }

                }
                
                return result.View(final_shape, inline:true);
            }
            
        }

        public FloatTensor ShapeAsTensor()
        {
            var data = new float[shape.Length];
            var ndims = new int[1];
            ndims[0] = shape.Length;
	        
            for (var dim = 0; dim < shape.Length; dim++)
            {
                data[dim] = shape[dim];
            }

            var result = factory.Create(_data: data, _shape: ndims);

            return result;
        }  
        
        public FloatTensor Sigmoid(bool inline = false, FloatTensor result = null)
        {
            if (dataOnGpu)
            {
                if (!inline) return SigmoidGPU(this.emptyTensorCopy());
                if (autograd)
                    throw new InvalidOperationException(
                        "Cannot call inline functions if you intend to run backprop.");

                SigmoidGPU_();
                return this;
            }
            
            result = HookGraph(ref result, "sigmoid", inline);

            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    if (this[i] >= 0)
                    {
                        var s = Math.Exp(-(double) this[i]);
                        result[i] = (float) (1 / (1.0f + s));
                    }
                    else
                    {
                        var s = Math.Exp((double) this[i]);
                        result[i] = (float) (s / (1.0f + s));
                    }
                }
            });

            return result;
        }
        
        public FloatTensor Sign(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return SignGPU(result);
                SignGPU_();
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Sign(x)).ToArray();
            return result;
        }

        public FloatTensor Sin(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return SinGPU();
                SinGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Sin((double) x)).ToArray();
            return result;
        }
        
        public FloatTensor Sinh(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return SinhGPU();
                SinhGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Sinh((double) x)).ToArray();
            return result;
        }

        public FloatTensor Std(int dim = -1, bool unbiased=true)
        {
            FloatTensor avg;
            
            if (dim == -1)
            {
                var mean = Mean(dim: dim);
                var diff = this.Sub(mean);
                var sqdiff = diff.Pow(2);
                if (unbiased)
                {
                    avg = sqdiff.Sum().Div(shape[0] - 1); // Bessel's correction
                }
                else
                {
                    avg = sqdiff.Mean(); 
                }
                var result = avg.Sqrt();
                return result;
            }
            else
            {
                var mean = Mean(dim:dim, keepdim:true);
                var diff = this.Sub(mean);
                var sqdiff = diff.Pow(2);
                
                if(unbiased)
                {
                    avg = sqdiff.Sum(dim).Div(shape[0] - 1); // Bessel's correction
                }
                else
                {
                    avg = sqdiff.Mean(dim); 
                }
                var result = avg.Sqrt();
                return result;
            }
            
        }
        
        // TODO: Softmax will run on GPU, when below OPS have a GPU implementation!
        // TODO: Improve the implementation!!!
        public FloatTensor Softmax(int dim = -1)
        {
            FloatTensor input = this;
            
            if (!input.IsContiguous())
                throw new NotImplementedException(
                    "Softmax Gradient does not support non-contiguous tensors at the moment!");

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

            var output = input.Copy();


            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = (outerSize * innerSize) * (workerId + 1) / nCpu;
                for (var i = (outerSize * innerSize) * workerId / nCpu; i < max; i++)
                {
                    int outerIdx = i / innerSize;
                    int innerIdx = i % innerSize;

                    // works for contiguous!!
                    var index = outerIdx * outerStride + innerIdx;

                    var inputMax = float.MinValue;
                    for (var d = 0; d < dimSize; d++)
                    {
                        if (output.Data[d * dimStride] >= inputMax)
                            inputMax = output.Data[d * dimStride];
                    }

                    float sum = 0;
                    for (var d = 0; d < dimSize; d++)
                    {
                        var z = (float) Math.Exp(output.Data[index + d * dimStride] - inputMax);
                        output.Data[index + d * dimStride] = z;
                        sum += z;
                    }

                    float invSum = 1 / sum;
                    for (var d = 0; d < dimSize; d++)
                    {
                        output.Data[index + d * dimStride] = output.Data[index + d * dimStride] * invSum;
                    }
                }
            });

            if (gpu)
            {
                output.Gpu(input.Shader);
            }

            output = input.HookGraph(ref output, "softmax-" + _dim.ToString(), false);

            return output;
        }

        public FloatTensor Sqrt(bool inline = false)
        {
            var result = inline ? this : this.emptyTensorCopy();

            if (dataOnGpu)
            {
                if (!inline) return SqrtGPU();
                SqrtGPU_();
                return this;
            }

            result.Data = data.AsParallel().Select(x => (float) Math.Sqrt((double) x)).ToArray();
            return result;
        }
        
        public FloatTensor Squeeze(int dim = -1, bool inline = false)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            var list = new List<int>();

            if (dim >= 0)
            {
                for (int i = 0; i < shape.Length; i++)
                {
                    if (i != dim)
                    {
                        list.Add(shape[i]);
                    }
                    else
                    {
                        if (shape[i] != 1)
                        {
                            list.Add(shape[i]);
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < shape.Length; i++)
                {
                    if (shape[i] > 1)
                    {
                        list.Add(shape[i]);
                    }
                }
            }

            FloatTensor result = this;
            
            if (list.Count == 0)
            {
                if (!inline)
                {
                    result = factory.Create(_data: data, _shape: shape, _shader: shader, _copyData: false);
                }
            }
            else
            {
                if (inline)
                {
                    View(list.ToArray(), inline: true);
                }
                else
                {
                    result = View(list.ToArray());
                }
            }

            return result;
        }        

        public FloatTensor Sub(FloatTensor x, bool inline = false, FloatTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");
            }

            // Check if both tensors are compatible for sub - fallback to scalar version if either tensor's size == 1
            if (SameSizeDimensionsShapeAndLocation(ref x))
            {
                if (x.Size == 1)
                {
                    return this.Sub(x.Expand(shape).Contiguous(), inline);
                }
                else if (this.Size == 1)
                {
                    if (inline)
                    {
                        throw new InvalidOperationException("Tensor sizes don't match");
                    }

                    return x.Sub(this.Expand(x.shape).Contiguous()).Neg();
                }
                else
                {
                    throw new InvalidOperationException();
                }
            }
            
            result = HookGraph(ref result, ref x, "sub_elem", inline);
            
            if (dataOnGpu & x.dataOnGpu)
            {
                if (inline)
                {
                    if (autograd)
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    SubElemGPU_(x);
                    return this;
                }
                result = SubElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a - b).ToArray();

            }

            return result;
        }  
        
        public FloatTensor Sub(float value, bool inline = false, FloatTensor result = null)
        {
            result = HookGraph (ref result, value, "sub_scalar", inline);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return SubScalarGPU(value, result);
                SubScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => x - value).ToArray();
            return result;
        }

        public FloatTensor Sum(int dim = -1, bool keepdim = false)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous, call Contiguous() to convert");
            }

            // TODO: Implement GPU op. with GPU tests.

            return Reduce(dim, keepdim, (acc, val, index, arr) => acc + val, (val, len) => val, creation_op:"sum_"+dim);

        }        
        
        public FloatTensor Tan(bool inline = false)
        {
            if (dataOnGpu)
            {
                if (!inline) return TanGPU();
                TanGPU_();
                return this;
            }
            var result = inline ? this : this.emptyTensorCopy();
            result.Data = data.AsParallel().Select(x => (float) Math.Tan((double) x)).ToArray();
            return result;
        }

        public FloatTensor Tanh(bool inline = false, FloatTensor result = null)
        {
            if (dataOnGpu)
            {
                return TanhGPU();
            }

            result = HookGraph(ref result, "tanh", inline);

            result.Data = data.AsParallel().Select(x => (float) Math.Tanh((double) x)).ToArray();
            return result;
        }

        public FloatTensor Transpose()
        {
            if (shape.Length != 2)
                throw new InvalidOperationException("Need to specify parameters for tensors with more than 2 dims.");

            return Transpose(0, 1);
        }

        public FloatTensor Transpose(int dimension1, int dimension2, FloatTensor result = null)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            //TODO: Should we create a new Tensor object here?
            if (dimension1 < 0 || dimension1 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension1");
            if (dimension2 < 0 || dimension2 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension2");

            if (dimension1 == dimension2)
            {
                return this;
            }

            var newShape = (int[]) Shape.Clone();
            var tmpDim = newShape[dimension1];
            newShape[dimension1] = newShape[dimension2];
            newShape[dimension2] = tmpDim;

            //var result = new FloatTensor(_controller: controller, _shape: newShape, _shader: this.shader);
            result = HookGraph(ref result, "transpose", false, newShape);
  
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    var idxs = GetIndices(i);
                    var tmp = idxs[dimension1];
                    idxs[dimension1] = idxs[dimension2];
                    idxs[dimension2] = tmp;
                    result[idxs] = this[i];
                }
            });

            return result;
        }
        
        public void Triu_(int k)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            if (shape.Length != 2)
            {
                throw new InvalidOperationException(
                    String.Format("Matrix multiply not possible: Num. Dimensions {0} != 2.", shape.Length));
            }
            if (dataOnGpu)
            {
                //UnityEngine.Debug.Log ("Entra");
                TriuGPU_(k);
                return;
            }
            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    var col = i % this.shape[1];
                    var row = (i - col) / this.shape[1];
                    if (col < row + k)
                    {
                        this[i] = 0.0f;
                    }
                }
            });
        }

        public FloatTensor Trunc(bool inline = false)
        {
            if (dataOnGpu)
            {
                return TruncGPU();
            }
            var result = factory.Create(_shape: shape, _shader: this.shader);
            result.Data = data.AsParallel().Select(x => (float) Math.Truncate((double) x)).ToArray();
            return result;
        }

        public float Trace()
        {
            if ((shape.Length != 2) || (shape[0] != shape[1]))
                throw new InvalidOperationException("Trace is defined on square 2d matrices only.");

            var stride = strides[0] + strides[1];
            return dataOnGpu
                ? TraceGPU()
                : Enumerable.Range(0, shape.Min()).AsParallel().Select(i => this[i * stride]).Sum();
        }

        public FloatTensor Unsqueeze(int dim, bool inline = false)
        {
            int[] new_shape = new int[shape.Length + 1];
            int j = 0;
            for (int i = 0; i < new_shape.Length; i++)
            {
                if (i == dim)
                {
                    new_shape[i] = 1;
                }
                else
                {
                    new_shape[i] = shape[j];
                    j += 1;
                }
            }

            return View(new_shape, inline:inline);
        }        
        
        public FloatTensor View(int[] new_shape, bool inline = false, FloatTensor result = null)
        {
            if (!IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }
            if (inline == true)
            {
                
                this.Shape = new_shape;

                if (dataOnGpu)
                {
                    shapeBuffer.Release();
                    shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                    shapeBuffer.SetData(shape);
                    
                }
                
                setStridesAndCheckShape();

                return this;

            }
            else
            {
                
                string shape_str = "";
                for (int i = 0; i < new_shape.Length; i++) shape_str += "_" + new_shape[i];
                result = HookGraph(ref result, creation_op:"view"+shape_str, inline:inline, resultShape:new_shape);
                result.Add(this, inline: true, override_checks:true);

                return result;

            }
            
        }

        public FloatTensor ViewAs(FloatTensor x, bool inline = false)
        {
            return this.View(x.shape, inline);
        }

// closes class and namespace
    }
}
