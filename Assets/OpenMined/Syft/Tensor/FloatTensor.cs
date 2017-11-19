using UnityEngine;
using System;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private float[] data;
        private long[] strides;
        private int[] shape;
        private int size;

//		private bool dataOnGpu;
//		public bool DataOnGpu => dataOnGpu;

        private long GetIndex(params int[] indices)
        {
            long offset = 0;
            for (int i = 0; i < indices.Length; ++i)
            {
                if(indices[i] >= shape[i] || indices[i] < 0)
                    throw new IndexOutOfRangeException();
                offset += indices[i] * strides[i];
            }
            return offset;
        }

        public float[] Data
        {
            get { return data; }
            
            set { data = value;  }
        }

        public int[] Shape
        {
            get { return shape; }
        }

        public int Size
        {
            get { return size; }
        }

        public FloatTensor(float[] _data, int[] _shape)
        {
            //TODO: Can contigous allocation might be a problem?
            //TODO: Should we create different allocation methods for CPU and GPU?

            this.size = _data.Length;
            this.strides = new long[_shape.Length];

            long acc = 1;
            for (int i = _shape.Length - 1; i >= 0; --i)
            {
                this.strides[i] = acc;
                acc *= _shape[i];
            }

            if (acc != this.size)
                throw new FormatException("Tensor shape and data do not match");

            this.data = (float[])_data.Clone();
            this.shape = _shape;
        }
        
        public float this[params int[] indices]
        {
            get
            {
                return data[GetIndex(indices)];
            }
            set
            {
                data[GetIndex(indices)] = value;
            }
        }
             
        public string Print()
        {
            if (dataOnGpu)
            {
                CopyGpuToCpu();
            }

            string print = "";

            int d1 = shape[0];
            int d2 = 1;
            if (shape.Length > 1)
                d2 = shape[1];
            int d3 = 1;
            if (shape.Length > 2)
                d3 = shape[2];
            if (shape.Length > 3)
                print += "Only printing first layer in dimesnions >= 4\n";
                
            for (int k = 0; k < d3; k++)
            {
                for (int i = 0; i < d1; i++)
                {
                    for (int j = 0; j < d2; j++)
                    {
                        float f = data[i * d2 + j + k * d1 * d2];                            
                        print += f.ToString() + "\t";
                    }
                    if(i < d1 - 1)
                        print += "\n\n";
                }
                if(k < d3 - 1)
                    print += "\n\n-----------\n\n";
            }
            return print;
        }
    }
}