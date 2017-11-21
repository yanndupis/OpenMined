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

			tensor.Transpose();

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
        public void TensorId()
        {
            float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            int[] shape = { 2, 3, 2 };

            var tensor1 = new FloatTensor(data, shape);
            var tensor2 = new FloatTensor(data, shape);
            
            Assert.AreNotEqual(tensor1.Id, tensor2.Id);
            Assert.AreEqual(tensor1.Id + 1, tensor2.Id);
        }
        
    }
}