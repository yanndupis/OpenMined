using System;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        // Should we put a check incase this variable overflows?
        private static volatile int _nCreated = 0;

        private float[] data;
        private long[] strides;
        private int[] shape;
        private int size;
        
        private int id;
        
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
        }

        public int[] Shape
        {
            get { return shape; }
        }

        public int Size
        {
            get { return size; }
        }

        public int Id
        {
            get { return id; }
            
            set { id = value; }
        }

        public FloatTensor(int[] _shape, bool _initOnGpu) {

            this.shape = (int[])_shape.Clone ();
            this.size = _shape[0];

            for (int i = 1; i < _shape.Length; i++) {
                this.size *= _shape [i];
            }

            this.data = new float[this.size];

            if (_initOnGpu) {
                Gpu ();
            }
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
            this.shape = (int[])_shape.Clone();
            
            // IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.
            this.Id = System.Threading.Interlocked.Increment(ref _nCreated); 
        }


        public float this[params int[] indices]
        {
            get
            {
                return Data[GetIndex(indices)];
            }
            set
            {
                Data[GetIndex(indices)] = value;
            }
        }
        
        
        public string Print()
        {
            if (dataOnGpu)
            {
                CopyGpuToCpu();
            }

            string print = "";

            if (shape.Length > 3)
                print += "Only printing the last 3 dimesnions\n";
            int d3 = 1;
            if (shape.Length > 2)
                d3 = shape[shape.Length - 3];
            int d2 = 1;
            if (shape.Length > 1)
                d2 = shape[shape.Length-2];
            int d1 = shape[shape.Length-1];

            for (int k = 0; k < d3; k++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int i = 0; i < d1; i++)
                    {
                        float f = data[i + j * d2 + k * d1 * d2 ];
                        print += f.ToString() + ",\t";
                    }
                    print += "\n";
                }
                print += "\n";
            }
            return print;
        }
    }
}