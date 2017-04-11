using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.Data2D;
using NNSharp.SequentialBased.SequentialLayers;

namespace UnitTests
{
    [TestClass]
    public class TestDense2D
    {
        [TestMethod]
        public void Test_Dense2D_Execute()
        {
            // Initialize data.
            Data2D data = new Data2D(3, 3, 2, 1);
            data[0, 0, 0, 0] = 1;
            data[1, 0, 0, 0] = 2;
            data[2, 0, 0, 0] = 0;

            data[0, 1, 0, 0] = 3;
            data[1, 1, 0, 0] = 4;
            data[2, 1, 0, 0] = 0;

            data[0, 2, 0, 0] = 2;
            data[1, 2, 0, 0] = 2;
            data[2, 2, 0, 0] = 0;


            data[0, 0, 1, 0] = 0;
            data[1, 0, 1, 0] = 3;
            data[2, 0, 1, 0] = 1;

            data[0, 1, 1, 0] = 1;
            data[1, 1, 1, 0] = 1;
            data[2, 1, 1, 0] = 1;

            data[0, 2, 1, 0] = 3;
            data[1, 2, 1, 0] = 1;
            data[2, 2, 1, 0] = 0;

            // Initialize weights.
            Data2D weights = new Data2D(3, 3, 2, 2);
            weights[0, 0, 0, 0] = 1;
            weights[1, 0, 0, 0] = 2;
            weights[2, 0, 0, 0] = 2;

            weights[0, 1, 0, 0] = 3;
            weights[1, 1, 0, 0] = 1;
            weights[2, 1, 0, 0] = 1;

            weights[0, 2, 0, 0] = 1;
            weights[1, 2, 0, 0] = 3;
            weights[2, 2, 0, 0] = 2;


            weights[0, 0, 1, 0] = 0;
            weights[1, 0, 1, 0] = 1;
            weights[2, 0, 1, 0] = 1;

            weights[0, 1, 1, 0] = 2;
            weights[1, 1, 1, 0] = 3;
            weights[2, 1, 1, 0] = 0;

            weights[0, 2, 1, 0] = 2;
            weights[1, 2, 1, 0] = 3;
            weights[2, 2, 1, 0] = 2;



            weights[0, 0, 0, 1] = 5;
            weights[1, 0, 0, 1] = 2;
            weights[2, 0, 0, 1] = 1;

            weights[0, 1, 0, 1] = 3;
            weights[1, 1, 0, 1] = 8;
            weights[2, 1, 0, 1] = 1;

            weights[0, 2, 0, 1] = 1;
            weights[1, 2, 0, 1] = 0;
            weights[2, 2, 0, 1] = 2;


            weights[0, 0, 1, 1] = 0;
            weights[1, 0, 1, 1] = 1;
            weights[2, 0, 1, 1] = 0;

            weights[0, 1, 1, 1] = 2;
            weights[1, 1, 1, 1] = 4;
            weights[2, 1, 1, 1] = 0;

            weights[0, 2, 1, 1] = 2;
            weights[1, 2, 1, 1] = 3;
            weights[2, 2, 1, 1] = 2;

            Dense2DLayer dens = new Dense2DLayer(weights);
            dens.SetInput(data);
            dens.Execute();
            Data2D output = dens.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 44.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], 70.0, 0.0000001);

        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullDense_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(3, 3, 3, 3);
            Dense2DLayer dens = new Dense2DLayer(weights);
            dens.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullConv_Weights()
        {
            Data2D weights = null;
            Dense2DLayer dens = new Dense2DLayer(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(3, 3, 3, 3);
            Dense2DLayer dens = new Dense2DLayer(weights);
            dens.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Weights()
        {
            DataArray weights = new DataArray(5);
            Dense2DLayer dens = new Dense2DLayer(weights);
        }
    }
}
