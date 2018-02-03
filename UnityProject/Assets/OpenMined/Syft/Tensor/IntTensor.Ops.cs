using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor
    {
        public IntTensor Add(int value, bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            IntTensor result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => x + value).ToArray();

            return result;
        }

        public FloatTensor Acos(bool inline = false)
        {
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(this.shape);

            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            result.Data = data.AsParallel().Select(x => (float)Math.Acos((double)x)).ToArray();

            return result;
        }

        public IntTensor Add(IntTensor x, bool inline = false)
        {

            IntTensor result = factory.Create(this.shape);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { AddElemGPU_(x); return this; }
                else { return AddElemGPU(x, result); }
            }
            else
            {
                // run Addition on the CPU
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a + b).ToArray();

                return result;
            }

        }

        public IntTensor Reciprocal(bool inline = false)
        {
            IntTensor result = factory.Create(this.shape);
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { ReciprocalGPU_(); return this; }
                else { return ReciprocalGPU(result); }
            }
            if (inline)
            {
                this.Data = data.AsParallel().Select(x => (int)(1 / x)).ToArray();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (int)(1 / x)).ToArray();
            return result;
        }

        public FloatTensor Sin(bool inline = false)
        {
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(shape);
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { throw new NotImplementedException(); }
                else { return SinGPU(result); }
            }
            result.Data = data.AsParallel().Select(x => (float)Math.Sin((double)x)).ToArray();
            return result;
        }

        public FloatTensor Cos(bool inline = false)
        {
            FloatTensor result = factory.ctrl.floatTensorFactory.Create(shape);
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { throw new NotImplementedException(); }
                else { return CosGPU(result); }
            }
            result.Data = data.AsParallel().Select(x => (float)Math.Cos((float)x)).ToArray();
            return result;
        }

        public IntTensor Eq(IntTensor other, bool inline = false)
        {
            // Run argument checks on CPU
            // Assuming broadcasting is not supported
            if (!this.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor dimensions must match");

            if (other == null)
                throw new ArgumentNullException();

            IntTensor result;

            if (dataOnGpu) {
              if (!inline){
                result = factory.Create(this.shape);

                // assign the gpu
                result.Gpu(shader);

                // find the id corresponding to the operation
                int kernel_id = shader.FindKernel("EqElemInt");

                // associate arrays with gpu
                shader.SetBuffer(kernel_id, "EqElemIntDataSelf", this.DataBuffer);
                shader.SetBuffer(kernel_id, "EqElemIntDataOther", other.DataBuffer);
                shader.SetBuffer(kernel_id, "EqElemIntDataResult", result.DataBuffer);

                // launch kernel
                shader.Dispatch(kernel_id, this.size, 1, 1);

                return result;
              }
              else {
                int kernel_id = shader.FindKernel("EqElemInt_");

                // associate resources
                shader.SetBuffer(kernel_id, "EqElemIntDataSelf_", this.DataBuffer);
                shader.SetBuffer(kernel_id, "EqElemIntDataOther_", other.DataBuffer);

                // launch kernel
                shader.Dispatch(kernel_id, this.size, 1, 1);

                return this;
              }
            }
            else {
                if (inline) {
                    this.Data = data.AsParallel().Zip(other.Data.AsParallel(),
                                                        (a, b) => a == b ? 1 : 0).ToArray();
                    return this;
                }
                else {
                    result = factory.Create(this.shape);
                    result.Data = data.AsParallel().Zip( other.Data.AsParallel(),
                                                        (a, b) => a == b ? 1 : 0 ).ToArray();
                    return result;
                }
            }
        }


        public IntTensor View(int[] new_shape, bool inline = true)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous, call Contiguous() to convert");
            }
            // suppport for -1 parameter value in new_shape
            var index = Array.IndexOf(new_shape, -1);
            if(index != -1)
            {
                int tempSize = 1;
                foreach(var s in new_shape)
                {
                    if (s != -1)
                        tempSize *= s;
                }
                new_shape[index] = size / tempSize;
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
                IntTensor result = factory.Create(new_shape);
                result.Add(this, inline: true);
                return result;
            }

        }

        public IntTensor Abs(bool inline = false)
        {
            IntTensor result = factory.Create(this.shape);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { AbsGPU_(); return this; }
                else { return AbsGPU(result); }
            }

            if (inline)
            {
                this.Data = data.AsParallel().Select(x => Math.Abs(x)).ToArray();
                return this;
            }
            else
            {
                result.Data = data.AsParallel().Select(x => Math.Abs(x)).ToArray();
                return result;
            }
        }

        public IntTensor Lt(IntTensor other, bool inline = false)
        {
            // Run argument checks on CPU anyway just to make sure
            if (!this.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor dimensions must match");

            if (other == null)
                throw new ArgumentNullException();

            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            else
            {
                if (inline)
                {
                    this.Data = data.AsParallel().Zip(other.Data.AsParallel(),
                                                        (a, b) => a < b ? 1 : 0).ToArray();
                    return this;
                }
                else
                {
                    IntTensor result = factory.Create(this.shape);
                    result.Data = data.AsParallel().Zip(other.Data.AsParallel(),
                                                        (a, b) => a < b ? 1 : 0).ToArray();
                    return result;
                }
            }
        }

        public IntTensor Sign(bool inline = false)
        {
            IntTensor result = factory.Create(this.shape);
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            if (!inline)
            {
                result.Data = data.AsParallel().Select(x => (int)Math.Abs(x) / x).ToArray();
            }
            return result;
        }

        public IntTensor Sqrt(bool inline = false)
        {

            if (dataOnGpu)
            {
                return this;
            }

            IntTensor result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => (int)Math.Sqrt(x)).ToArray();

            return result;
        }

        public IntTensor Neg(bool inline = false, IntTensor result = null)
        {
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { NegGPU_(); return this; }
                else { result = factory.Create(this.shape); return NegGPU(result); }
            }
            result = this;
            if (!inline) result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => -x).ToArray();
            return result;
        }

        public IntTensor Transpose(bool inline = false)
        {
            if (shape.Length != 2)
            {
                throw new InvalidOperationException("Need to specify parameters for tensors with more than 2 dims.");
            }
            return Transpose(0, 1, inline: inline);
        }

        public IntTensor Transpose(int dimension1, int dimension2, IntTensor result = null, bool inline = false)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous, call Contiguous() to convert");
            }

            if (dimension1 < 0 || dimension1 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension1");
            if (dimension2 < 0 || dimension2 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension2");

            if (dimension1 == dimension2)
            {
                return this;
            }

            var newShape = (int[])shape.Clone();
            var tmpDimension = newShape[dimension1];
            newShape[dimension1] = newShape[dimension2];
            newShape[dimension2] = tmpDimension;

            result = factory.Create(newShape);

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

        public bool Equal(IntTensor x, bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            return this.Shape.SequenceEqual(x.Shape) && data.AsParallel().SequenceEqual(x.Data.AsParallel());
        }

        public IntTensor Sub(IntTensor x, bool inline = false)
        {

            IntTensor result = factory.Create(this.shape);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { SubGPU_(x); return this; }
                else { return SubGPU(x, result); }
            }
            else
            {
                result = inline ? this : factory.Create(this.shape);
                // run Subtraction on the CPU
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a - b).ToArray();

                return result;
            }

        }

        public IntTensor Pow(IntTensor x, bool inline = false, IntTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous())
            {
                throw new InvalidOperationException("All tensors must be contiguous, call Contiguous() to convert");
            }

            result = inline ? this : factory.Create(this.shape);

            result.Data = data.AsParallel().Zip(
              x.Data.AsParallel(),
              (a, b) => (int)Math.Pow((int)a, b)
            ).ToArray();

            return result;
        }

        public IntTensor Pow(int value, bool inline = false, IntTensor result = null)
        {
            result = inline ? this : factory.Create(this.shape);

            result.Data = data.AsParallel().Select(x => (int)Math.Pow(x, value)).ToArray();

            return result;
        }

        public IntTensor Sub(int value, bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            IntTensor result = inline ? this : factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => x - value).ToArray();

            return result;
        }

        public IntTensor Tan(bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            IntTensor result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => (int)Math.Tan((int)x)).ToArray();
            return result;
        }

        public int Trace()
        {
            if ((shape.Length != 2) || (shape[0] != shape[1]))
                throw new InvalidOperationException("Trace is defined on square 2d matrices only.");

            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            var stride = strides[0] + strides[1];
            return Enumerable.Range(0, shape.Min()).AsParallel().Select(i => this[i * stride]).Sum();
        }

        // closes class and namespace
    }
}
