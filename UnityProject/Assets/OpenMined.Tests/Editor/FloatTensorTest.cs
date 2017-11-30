using System;
using System.Diagnostics;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;

using OpenMined.Syft.Tensor;

namespace OpenMined.Tests
{

    public class FloatTensorTest
    {

        [TestFixtureSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
        }

        [TestFixtureTearDown]
        public void CleanUp()
        {
            //CleanUp runs once after all test cases are finished.
        }

        [SetUp]
        public void SetUp()
        {
            //SetUp runs before all test cases
        }

        [TearDown]
        public void TearDown()
        {
            //SetUp runs after all test cases
        }

        [Test]
        public void Copy()
        {
            float[] array = { 1, 2, 3, 4, 5 };
            int[] shape = { 5 };

            var tensor = new FloatTensor(array, shape);
            var copy = tensor.Copy();

            Assert.AreEqual(copy.Shape,tensor.Shape);
            Assert.AreEqual(copy.Data, tensor.Data);
            Assert.AreNotEqual(copy.Id, tensor.Id);
        }

        [Test]
        public void Create1DTensor()
        {
            float[] array = { 1, 2, 3, 4, 5 };
            int[] shape = { 5 };

            var tensor = new FloatTensor(array, shape);

            Assert.AreEqual(array.Length, tensor.Size);

            for (int i = 0; i < array.Length; i++)
            {
                Assert.AreEqual(array[i], tensor[i]);
            }
        }

        [Test]
        public void Create2DTensor()
        {
            float[,] array = { { 1, 2, 3, 4, 5 }, { 6, 7, 8, 9, 10 } };

            float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape = { 2, 5 };

            var tensor = new FloatTensor(data, shape);

            Assert.AreEqual(array.GetLength(0), tensor.Shape[0]);
            Assert.AreEqual(array.GetLength(1), tensor.Shape[1]);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Assert.AreEqual(array[i, j], tensor[i, j]);
                }
            }
        }

        [Test]
        public void Create3DTensor()
        {
            float[,,] array = { { { 1, 2 }, { 3, 4 }, { 5, 6 } }, { { 7, 8 }, { 9, 10 }, { 11, 12 } } };

            float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape = { 2, 3, 2 };

            var tensor = new FloatTensor(data, shape);

            Assert.AreEqual(array.GetLength(0), tensor.Shape[0]);
            Assert.AreEqual(array.GetLength(1), tensor.Shape[1]);
            Assert.AreEqual(array.GetLength(2), tensor.Shape[2]);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    for (int k = 0; k < array.GetLength(2); k++)
                    {
                        Assert.AreEqual(array[i, j, k], tensor[i, j, k]);
                    }
                }
            }
        }

        [Test]
        public void Transpose2D()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5};

            float[] data2 = { 1, 6, 2, 7, 3, 8, 4, 9, 5, 10 };
            int[] shape2 = { 5, 2 };

            var tensor = new FloatTensor(data1, shape1);
            var transpose = new FloatTensor(data2, shape2);

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    Assert.AreEqual(tensor[i, j], transpose[j, i]);
                }
            }

            tensor.Transpose();

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    Assert.AreEqual(tensor[i, j], transpose[i, j]);
                }
            }
        }

        [Test]
        public void Transpose3D()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape1 = { 3, 2, 2};

            float[] data2 = { 1, 5, 9, 3, 7, 11, 2, 6, 10, 4, 8, 12 };
            int[] shape2 = { 2, 2, 3 };

            var tensor = new FloatTensor(data1, shape1);
            var transpose = new FloatTensor(data2, shape2);

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    for (int k = 0; k < tensor.Shape [2]; k++)
                    {
                        Assert.AreEqual (tensor [i, j, k], transpose [k, j, i]);
                    }
                }
            }
            tensor.Transpose(0, 2);

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    for (int k = 0; k < tensor.Shape [2]; k++)
                    {
                        Assert.AreEqual(tensor[i, j, k], transpose[i, j, k]);
                    }
                }
            }
        }

        [Test]
        public void TransposeNoDimensionsSpecified()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape1 = { 3, 2, 2 };

            // Test Tensor with more than 2 dimensions
            var tensor = new FloatTensor(data1, shape1);
            Assert.That(() => tensor.Transpose(),
                Throws.TypeOf<InvalidOperationException>());

            // Test tensor with less than 2 dimensions
            float[] data2 = { 1, 2, 3, 4, 5 };
            int[] shape2 = { 5 };
            tensor = new FloatTensor(data2, shape2);
            Assert.That(() => tensor.Transpose(),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void TransposeDimensionsOutOfRange()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape1 = {3, 2, 2};

            // Test negative dimension indexes
            var tensor = new FloatTensor(data1, shape1);
            Assert.That(() => tensor.Transpose(-1, 0),
                Throws.TypeOf<ArgumentOutOfRangeException>());
            Assert.That(() => tensor.Transpose(0, -1),
                Throws.TypeOf<ArgumentOutOfRangeException>());

            // Test dimension indexes bigger than tensor's shape lenght
            var tensor2 = new FloatTensor(data1, shape1);
            Assert.That(() => tensor2.Transpose(3, 0),
                Throws.TypeOf<ArgumentOutOfRangeException>());
            Assert.That(() => tensor2.Transpose(0, 3),
                Throws.TypeOf<ArgumentOutOfRangeException>());
        }

        [Test]
        public void Add()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
            var tensor2 = new FloatTensor(data2, shape2);

            var tensorSum = tensor1.Add (tensor2);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual (tensor1.Data [i] + tensor2.Data [i], tensorSum.Data [i]);
            }
        }

        [Test]
        public void AddUnequalSizes()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape2 = { 2, 6 };
            var tensor2 = new FloatTensor(data2, shape2);

            Assert.That(() => tensor1.Add(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalDimensions()
        {
            float[] data1 = { 1, 2, 3, 4 };
            int[] shape1 = { 4 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4 };
            int[] shape2 = { 2, 2 };
            var tensor2 = new FloatTensor(data2, shape2);

            Assert.That(() => tensor1.Add(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalShapes()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6 };
            int[] shape1 = { 2, 3 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4, 5, 6 };
            int[] shape2 = { 3, 2 };
            var tensor2 = new FloatTensor(data2, shape2);

            Assert.That(() => tensor1.Add(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void Abs_()
        {
            float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
            int[] shape1 = { 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 0, 1, float.MaxValue, -float.MinValue };
            int[] shape2 = { 5 };
            var tensorAbs = new FloatTensor(data2, shape2);

            tensor1.Abs_();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor1.Data[i], tensorAbs.Data[i]);
            }
        }

        [Test]
        public void Neg()
        {
            float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
            int[] shape1 = { 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 0, -1, -float.MaxValue, -float.MinValue };
            int[] shape2 = { 5 };
            var tensorNeg = new FloatTensor(data2, shape2);

            var result = tensor1.Neg ();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (result.Data[i], tensorNeg.Data[i]);
            }
        }

        [Test]
        public void Sigmoid_()
        {
            float[] data1 = { 0.0f };
            int[] shape1 = { 1 };
            var tensor1 = new FloatTensor(data1, shape1);
            tensor1.Sigmoid_();
            Assert.AreEqual(tensor1.Data[0], 0.5f);

            float[] data2 = { 0.1f, 0.5f, 1.0f, 2.0f };
            float[] data3 = { -0.1f, -0.5f, -1.0f, -2.0f };
            int[] shape2 = { 4 };
            var tensor2 = new FloatTensor(data2, shape2);
            var tensor3 = new FloatTensor(data3, shape2);
            tensor2.Sigmoid_();
            tensor3.Sigmoid_();
            var sum = tensor2.Add(tensor3);

            for (int i = 0; i < sum.Size; i++)
            {
                Assert.AreEqual(sum.Data[i], 1.0f);
            }
        }


        [Test]
        public void Zero_()
        {
            float[] data1 = { -1, 0, 1, float.MaxValue, float.MinValue };
            int[] shape1 = { 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 0, 0, 0, 0, 0 };
            int[] shape2 = { 5 };
            var tensorZero = new FloatTensor(data2, shape2);

            tensor1.Zero_ ();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor1.Data[i], tensorZero.Data[i]);
            }
        }

        [Test]
        public void ElementwiseMultiplication()
        {
            float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
            int[] shape1 = {2, 4};
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
            int[] shape2 = {2, 4};
            var tensor2 = new FloatTensor(data2, shape2);

            var tensorMult = tensor1.Mul (tensor2);

            for (int i = 0; i < tensorMult.Size; i++)
            {
                float current = tensor1.Data [i] * tensor2.Data [i];
                Assert.AreEqual (tensorMult.Data [i], current);
            }
        }

        [Test]
        public void ElementwiseMultiplicationUnequalSizes()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape2 = { 2, 6 };
            var tensor2 = new FloatTensor(data2, shape2);

			Assert.That(() => tensor1.Mul(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void ElementwiseMultiplicationUnequalDimensions()
        {
            float[] data1 = { 1, 2, 3, 4 };
            int[] shape1 = { 4 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4 };
            int[] shape2 = { 2, 2 };
            var tensor2 = new FloatTensor(data2, shape2);

			Assert.That(() => tensor1.Mul(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void ElementwiseMultiplicationUnequalShapes()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6 };
            int[] shape1 = { 2, 3 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4, 5, 6 };
            int[] shape2 = { 3, 2 };
            var tensor2 = new FloatTensor(data2, shape2);

			Assert.That(() => tensor1.Mul(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

//        [Test]
//        public void ElementwiseMultiplicationDataOnDifferent()
//        {
//            int[] shape1 = { 2, 3 };
//            var tensor1 = new FloatTensor(shape1);
//            int[] shape2 = { 2, 3 };
//            var tensor2 = new FloatTensor(shape2);
//
//			Assert.That(() => tensor1.Mul(tensor2),
//                Throws.TypeOf<InvalidOperationException>());
//        }

        [Test]
        public void TensorId()
        {
            float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape = { 2, 3, 2 };

            var tensor1 = new FloatTensor(data, shape);
            var tensor2 = new FloatTensor(data, shape);

            Assert.AreNotEqual(tensor1.Id, tensor2.Id);
            Assert.AreEqual(tensor1.Id + 1, tensor2.Id);
        }

        [Test]
        public void ScalarMultiplication()
        {
            float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
            int[] shape1 = {2, 4};
            var tensor1 = new FloatTensor(data1, shape1);
            var tensor2 = new FloatTensor(data1, shape1);

            // Test multiplication by 0
            float scalar = 0;
			var result = tensor1.Mul (scalar);
            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
            }

            // Test multiplication by positive
            tensor1 = new FloatTensor(data1, shape1);
            scalar = 99;
            result = tensor1.Mul (scalar);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
            }

            // Test multiplication by negative
            tensor1 = new FloatTensor(data1, shape1);
            scalar = -99;
            result = tensor1.Mul (scalar);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
            }

            // Test multiplication by decimal
            tensor1 = new FloatTensor(data1, shape1);
            scalar = 0.000001f;
            result = tensor1.Mul (scalar);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] * scalar, result.Data [i] );
            }
        }
        [Test]
        public void Ceil()
        {
            float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
            int[] shape1 = { 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 6, -20, 9, 101, 101 };
            int[] shape2 = { 5 };
            var tensorCeil = new FloatTensor(data2, shape2);

            var result = tensor1.Ceil ();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (result.Data[i], tensorCeil.Data[i]);
            }
        }

        [Test]
        public void Floor_()
        {
            float[] data1 = { 5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f };
            int[] shape1 = { 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 5, -21, 9, 100, 100 };
            int[] shape2 = { 5 };
            var tensorFloor = new FloatTensor(data2, shape2);

            tensor1.Floor_();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor1.Data[i], tensorFloor.Data[i]);
            }
        }

        [Test]
        public void Add_()
        {
            float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
            int[] shape1 = {2, 4};
            var tensor1 = new FloatTensor(data1, shape1);
            var tensor2 = new FloatTensor(data1, shape1);

            // Test addition by 0
            float val = 0;
            tensor1.Add_ (val);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] + val, tensor1.Data [i] );
            }

            // Test addition by positive
            tensor1 = new FloatTensor(data1, shape1);
            val = 99;
            tensor1.Add_ (val);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] + val, tensor1.Data [i] );
            }

            // Test addition by negative
            tensor1 = new FloatTensor(data1, shape1);
            val = -99;
            tensor1.Add_ (val);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] + val, tensor1.Data [i] );
            }

            // Test addition by decimal
            tensor1 = new FloatTensor(data1, shape1);
            val = 0.000001f;
            tensor1.Add_ (val);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual (tensor2.Data [i] + val, tensor1.Data [i] );
            }
        }

        [Test]
        public void ElementwiseSubtract()
        {
            float[] data1 = { float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue };
            int[] shape1 = {2, 4};
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue };
            int[] shape2 = {2, 4};
            var tensor2 = new FloatTensor(data2, shape2);

			var tensor = tensor1.Sub (tensor2);

            for (int i = 0; i < tensor.Size; i++)
            {
                float current = tensor1.Data [i] - tensor2.Data [i];
				Assert.AreEqual (current, tensor.Data [i]);
            }
        }

        [Test]
        public void ElementwiseSubtractUnequalSizes()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape2 = { 2, 6 };
            var tensor2 = new FloatTensor(data2, shape2);

			Assert.That(() => tensor1.Sub(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void ElementwiseSubtractUnequalDimensions()
        {
            float[] data1 = { 1, 2, 3, 4 };
            int[] shape1 = { 4 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4 };
            int[] shape2 = { 2, 2 };
            var tensor2 = new FloatTensor(data2, shape2);

			Assert.That(() => tensor1.Sub(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void ElementwiseSubtractUnequalShapes()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6 };
            int[] shape1 = { 2, 3 };
            var tensor1 = new FloatTensor(data1, shape1);

            float[] data2 = { 1, 2, 3, 4, 5, 6 };
            int[] shape2 = { 3, 2 };
            var tensor2 = new FloatTensor(data2, shape2);

			Assert.That(() => tensor1.Sub(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }
//
//        [Test]
//        public void ElementwiseSubtractDataOnDifferent()
//        {
//            int[] shape1 = { 2, 3 };
//            var tensor1 = new FloatTensor(shape1, true);
//            int[] shape2 = { 2, 3 };
//            var tensor2 = new FloatTensor(shape2, false);
//
//			Assert.That(() => tensor1.SubtractElementwise(tensor2),
//                Throws.TypeOf<InvalidOperationException>());
//        }
//
        [Test]
        public void AddMatrixMultiplyTest()
        {
            float[] base1_data = new float[] { 1, 2, 3, 4 };
            int[] base1_shape = new int[] { 2, 2 };
            var base1 = new FloatTensor(base1_data, base1_shape);

            float[] base2_data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            int[] base2_shape = new int[] { 3, 3 };
            var base2 = new FloatTensor( base2_data,base2_shape );

            float[] data = new float[] { 1, 2, 3, 4, 5, 6 };
            int[] tensor1_shape = new int[]{ 2, 3 };
            int[] tensor2_shape = new int[]{ 3, 2 };
            var tensor1 = new FloatTensor(data, tensor1_shape);
            var tensor2 = new FloatTensor(data, tensor2_shape);

            base1.AddMatrixMultiply(tensor1, tensor2);
            base2.AddMatrixMultiply(tensor2, tensor1);

            for (int i = 0; i < base1_shape[0]; i++)
            {
                for (int j = 0; j < base1_shape[1]; j++)
                {
                    float mm_res = base1_data[i * base1_shape[1] + j];
                    for (int k = 0; k < tensor1_shape[1]; k++)
                    {
                        mm_res += tensor1[i, k] * tensor2[k, j];
                    }
                    Assert.AreEqual(base1[i, j], mm_res);
                }
            }

            for (int i = 0; i < base2_shape[0]; i++)
            {
                for (int j = 0; j < base2_shape[1]; j++)
                {
                    float mm_res = base2_data[i * base2_shape[1] + j];
                    for (int k = 0; k < tensor2_shape[1]; k++)
                    {
                        mm_res += tensor2[i, k] * tensor1[k, j];
                    }
                    Assert.AreEqual(base2[i, j], mm_res);
                }
            }
        }

        [Test]
        public void Tanh()
        {
            float[] data1 = { -0.6366f, 0.2718f, 0.4469f, 1.3122f };
            int[] shape1 = { 4 };
            var tensor = new FloatTensor(data1, shape1);

            float[] data2 = { -0.562580109f, 0.265298963f, 0.419347495f, 0.86483103f };
            int[] shape2 = { 4 };
            var expectedTanhTensor = new FloatTensor(data2, shape2);

            var actualTanhTensor = tensor.Tanh();

            for (int i = 2; i < actualTanhTensor.Size; i++)
            {
                 Assert.AreEqual (expectedTanhTensor.Data[i], actualTanhTensor.Data[i]);
            }
        }

        [Test]
        public void IsContiguous()
        {
          float[] data = new float[] { 1, 2, 3, 4, 5, 6 };
          int[] shape = new int[]{ 2, 3 };
          var tensor = new FloatTensor(data, shape);
          Assert.AreEqual(tensor.IsContiguous(), true);
          var transposedTensor = tensor.Transpose();
          Assert.AreEqual(transposedTensor.IsContiguous(), false);
        }

        // TODO: AddMatrixMultiplyTests when implemented on CPU
        // TODO: MultiplyDerivative when implemented on CPU
    }
}
