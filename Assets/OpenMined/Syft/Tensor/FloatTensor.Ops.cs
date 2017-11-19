using UnityEngine;
using System;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private void SwapElements(ref int[] target, int index1, int index2)
        {
            int tmp = target[index1];
            target[index1] = target[index2];
            target[index2] = tmp;
        }

        private void SwapElements(ref long[] target, int index1, int index2)
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

        public FloatTensor Add(FloatTensor x)
        {
            FloatTensor output = new FloatTensor(this.shape, dataOnGpu);


            if (dataOnGpu)
            {
                // GPU Add Code Here
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    output.data[i] = x.Data[i] + this.data[i];
                }
            }

            return output;
        }

        public FloatTensor Abs()
        {
            if (dataOnGpu)
            {
                // GPU Absolute Value Code Here
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (data[i] < 0)
                    {
                        data[i] = -data[i];
                    }
                }
            }
            return this;
        }


        public FloatTensor Neg()
        {
            if (dataOnGpu)
            {
                NegGPU();
            }
            else
            {
                // run CPU code
                for (int i = 0; i < size; i++)
                {
                    data[i] = -data[i];
                }
            }
            return this;
        }

        public FloatTensor ParallelNeg()
        {
            int nCpu = SystemInfo.processorCount;
            var tasks = new Task[nCpu];
            for (int taskNumber = 0; taskNumber < nCpu; taskNumber++)
            {
                // capturing taskNumber in lambda wouldn't work correctly
                int taskNumberCopy = taskNumber;

                tasks[taskNumber] = Task.Factory.StartNew(
                    () =>
                    {
                        var max = data.Length * (taskNumberCopy + 1) / nCpu;
                        for (int i = data.Length * taskNumberCopy / nCpu;
                            i < max;
                            i++)
                        {
                            data[i] = -data[i];
                        }
                    });
            }
            Task.WaitAll(tasks);
            return this;
        }
        
        public FloatTensor ParallelBNeg()
        {
            int nCpu = SystemInfo.processorCount;
            Parallel.For(0, data.Length, i =>
            {
                data[i] = -data[i];
            });
            return this;
        }


        public FloatTensor ElementwiseMultiplication(FloatTensor other)
        {
            //TODO: make a better check. comparing size is not enough
            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    ElementwiseMultiplicationOnGpu(other);
                }
                else if (!dataOnGpu && !other.DataOnGpu)
                {
                    for (int i = 0; i < size; i++)
                    {
                        data[i] = data[i] * other.data[i];
                    }
                }
                else
                {
                    Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
                }
            }
            else
            {
                Debug.Log("Tensors do not have the same number of elements!");
            }
            return this;
        }

        public FloatTensor ScalarMultiplication(float scalar)
        {
            if (dataOnGpu)
            {
                ScalarMultiplicationOnGpu(scalar);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    data[i] = data[i] * scalar;
                }
            }
            return this;
        }

        public FloatTensor ElementwiseSubtract(FloatTensor other)
        {
            //Debug.LogFormat("<color=blue>FloatTensor.inline_elementwise_subtract dataOnGpu: {0}</color>", dataOnGpu);

            if (size == other.Size)
            {
                if (dataOnGpu && other.DataOnGpu)
                {
                    ElementwiseSubtractOnGpu(other);
                }
                else if (!dataOnGpu && !other.dataOnGpu)
                {
                    for (int i = 0; i < size; i++)
                    {
                        data[i] = data[i] - other.data[i];
                    }
                }
                else
                {
                    Debug.Log("Data for both Tensors needs to be colocated on the same device. - CPU != GPU");
                }
            }
            else
            {
                Debug.Log("Tensors do not have the same number of elements!");
            }
            return this;
        }

        public FloatTensor AddMatrixMultiply(FloatTensor tensor1, FloatTensor tensor2)
        {
            // TODO: check for corner cases

            bool gpu = dataOnGpu & tensor1.DataOnGpu & tensor2.DataOnGpu;
            bool cpu = !(dataOnGpu | tensor1.DataOnGpu | tensor2.DataOnGpu);

            if (gpu)
            {
                AddMatrixMultiplyOnGpu(tensor1, tensor2);
            }
            else if (cpu)
            {
                //TODO: implement the function
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }
            return this;
        }

        public FloatTensor MultiplyDerivative(FloatTensor other)
        {
            // TODO: check for corner cases
            if (dataOnGpu & other.DataOnGpu)
            {
                MultiplyDerivativeOnGpu(other);
            }
            else if (!dataOnGpu & !other.DataOnGpu)
            {
                //TODO: implement the function
            }
            else
            {
                Debug.Log("Data for all Tensors needs to be colocated on the same device. - CPU != GPU");
            }
            return this;
        }
    }
}