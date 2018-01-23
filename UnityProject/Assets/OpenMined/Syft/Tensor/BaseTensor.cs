﻿using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenMined.Network.Controllers;
using UnityEngine;
using OpenMined.Network.Utils;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;


namespace OpenMined.Syft.Tensor
{

    [Serializable]
    public abstract partial class BaseTensor<T>
    {
        #region Statics 

        // Should we put a check incase this variable overflows?
        protected static volatile int nCreated = 0;

        #endregion

        #region Members

        [SerializeField] protected T[] data;
        protected int id;
        [SerializeField] protected int[] strides;
        [SerializeField] protected int[] shape;
        [SerializeField] protected int size;

        protected ComputeShader shader;

        protected int usage_count;
        #endregion

        #region Properties

        public ComputeShader Shader
        {
            get { return shader; }
            set { shader = value; }
        }

        public T[] Data
        {
            get { return data; }
            set { data = value; }
        }

        public int[] Shape
        {
            get { return shape; }
            protected set { shape = value; }
        }

        public int[] Strides
        {
            get { return strides; }
            protected set { strides = value; }
        }

        public int Size
        {
            get { return size; }
            protected set { size = value; }
        }

        public int Usage_count
        {
            get { return usage_count; }
            protected set { usage_count = value; }
        }

        public int Id
        {
            get { return id; }
            protected set { id = value; }
        }

        public static int CreatedObjectCount
        {
            get { return nCreated; }
            protected set { nCreated = value; }
        }

        #endregion

        public void InitCpu(T[] _data, bool _copyData)
        {
            // looks like we have CPU data being passed in... initialize a CPU tensor.
            dataOnGpu = false;

            if (_copyData)
            {
                data = (T[]) _data.Clone();
            }
            else
            {
                data = _data;
            }
        }

        public void InitGpu(ComputeShader _shader, ComputeBuffer _dataBuffer, ComputeBuffer _shapeBuffer, ComputeBuffer _stridesBuffer,
            bool _copyData)
        {
            if (!SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("Shaders are not supported on the host machine");
            
            if (_shader != null)
            {
                shader = _shader;
            }
            else
            {
                throw new FormatException("You tried to initialize a GPU tensor without access to a shader or gpu.");
            }
            
            if (_copyData)
            {
                var tempData = new T[_dataBuffer.count];
                var tempShape = new int[shape.Length];
                var tempStrides = new int[shape.Length];

                _dataBuffer.GetData(tempData);
                _shapeBuffer.GetData(tempShape);
                _stridesBuffer.GetData(tempStrides);

                dataBuffer = new ComputeBuffer(_dataBuffer.count, Marshal.SizeOf(default(T)));
                shapeBuffer = new ComputeBuffer(_shapeBuffer.count, sizeof(int));
                stridesBuffer = new ComputeBuffer(_stridesBuffer.count, sizeof(int));

                dataBuffer.SetData(tempData);
                shapeBuffer.SetData(tempShape);
                stridesBuffer.SetData(tempStrides);
            }

            // Third: let's set the tensor's size to be equal to that of the buffer
            size = _dataBuffer.count;
        }

        #region Operators

        public int GetIndex(params int[] indices)
        {
            if (indices.Length < shape.Length)
                throw new NotSupportedException();
            var offset = 0;
            for (var i = 0; i < indices.Length; ++i)
            {
                if (indices[i] >= shape[i] || indices[i] < 0)
                    throw new IndexOutOfRangeException();
                offset += indices[i] * strides[i];
            }
            return offset;
        }

        public int[] GetIndices(int index)
        {
            var idx = index;
            var indices = new int[Shape.Length];
            for (var i = 0; i < Shape.Length; ++i)
            {
                indices[i] = (idx - (idx % (strides[i]))) / strides[i];
                idx -= indices[i] * strides[i];
            }
            return indices;
        }

        public T this[params int[] indices]
        {
            get { return this[GetIndex(indices)]; }
            set { this[GetIndex(indices)] = value; }
        }

        public T this[int index]
        {
            get { return Data[index]; }
            set { Data[index] = value; }
        }

        #endregion
        public void Zero_()
        {
            if (dataOnGpu)
            {
                ZeroGPU_();
                return;
            }

            Array.Clear(data, 0, size);
        }

        public void ZeroGPU_()
        {
            shader.SetBuffer(ZeroKernel_, "ZeroData_", dataBuffer);
            shader.Dispatch(ZeroKernel_, this.size, 1, 1);
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

        public int DimIndices2DataIndex(ref int[] dim_indices)
        {
            int index = 0;
            for (int i = 0; i < dim_indices.Length; i++)
            {
                index += dim_indices[i] * strides[i];
            }
            return index;
        }

        public int[] DataIndex2DimIndices(int index, ref int[] dim_indices)
        {
            if (dim_indices == null)
            {
                dim_indices = new int[strides.Length];
            }

            for (int i = 0; i < strides.Length; i++)
            {
                if (strides[i] != 0)
                {
                    dim_indices[i] = index / strides[i];
                    index %= strides[i];
                }
                else
                {
                    dim_indices[i] = 0;
                }
            }

            return dim_indices;
        }
        
        public abstract string ProcessMessage(Command msgObj, SyftController ctrl);

        public void setStridesAndCheckShape()
        {
            // Third: let's initialize our strides.
            strides = new int[shape.Length];

            // Fifth: we should check that the buffer's size matches our shape.
            int acc = 1;
            for (var i = shape.Length - 1; i >= 0; --i)
            {
                strides[i] = acc;
                acc *= shape[i];
            }

            // Sixth: let's check to see that our shape and data sizes match.
            size = acc;
        }

        public JToken GetConfig()
        {
            var jData = new JArray();
            for (var i = 0; i < data.Length; ++i)
            {
                jData.Add(data[i]);
            }

            var jStride = new JArray();
            for (var i = 0; i < strides.Length; ++i)
            {
                jStride.Add(strides[i]);
            }

            var jShape = new JArray();
            for (var i = 0; i < shape.Length; ++i)
            {
                jShape.Add(shape[i]);
            }

            return new JObject
            {
                { "data", jData },
                { "strides", jStride },
                { "shape", jShape },
                { "size", size }
            };
        }



    }
}