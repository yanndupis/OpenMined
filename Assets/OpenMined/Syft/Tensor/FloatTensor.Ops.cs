using System;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private	void SwapElements(ref int[] target, int index1, int index2)
        {
            int tmp = target[index1];
            target[index1] = target[index2];
            target[index2] = tmp;
        }
        
        private	void SwapElements(ref long[] target, int index1, int index2)
        {
            long tmp = target[index1];
            target[index1] = target[index2];
            target[index2] = tmp;
        }

        public FloatTensor Transpose()
        {
            if (shape.Length != 2)
                throw new InvalidOperationException("Need to specify parameters for tensors with more than 2 dims.");

            return Transpose(0, 1);
        }

        public FloatTensor Transpose(int dimension1, int dimension2)
        {
            //TODO: Should we create a new Tensor object here?
            
            if (dimension1 < 0 || dimension1 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension1");
            if (dimension2 < 0 || dimension2 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension2");

            if (dimension1 == dimension2)
            {
                return this;
            }

            SwapElements(ref strides, dimension1, dimension2);
            SwapElements(ref shape, dimension1, dimension2);

            return this;
        }

		public FloatTensor Abs() 
		{
			if (dataOnGpu) {
				// GPU Absolute Value Code Here
			} else {
				for (int i = 0; i < size; i++) {
					if (data [i] < 0) {
						data [i] = -data [i];
					} else {
						// do nothing
					}
				}
			}
			return this;
		}
    }
}