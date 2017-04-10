using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestSoftmax
    {
        [TestMethod]
        public void Test_Softmax_Execute()
        {

            // Softmax output
            softmax = new SoftmaxLayer();
            DataArray data = new DataArray(5);

            data[0] = 0.0;
            data[1] = 1.0;
            data[2] = 1.5;
            data[3] = 2.0;
            data[4] = 3.0;

            softmax.SetInput(data);
            softmax.Execute();

            DataArray output = softmax.GetOutput() as DataArray;

            // Expected output
            double[] expOu = new double[5];

            double sum = 0.0;
            sum += (Math.Exp(0.0) + Math.Exp(1.0) + Math.Exp(1.5) + Math.Exp(2.0) + Math.Exp(3.0));

            expOu[0] = Math.Exp(0.0) / sum;
            expOu[1] = Math.Exp(1.0) / sum;
            expOu[2] = Math.Exp(1.5) / sum;
            expOu[3] = Math.Exp(2.0) / sum;
            expOu[4] = Math.Exp(3.0) / sum;

            Assert.AreEqual(output[0], expOu[0], 0.00000001);
            Assert.AreEqual(output[1], expOu[1], 0.00000001);
            Assert.AreEqual(output[2], expOu[2], 0.00000001);
            Assert.AreEqual(output[3], expOu[3], 0.00000001);
            Assert.AreEqual(output[4], expOu[4], 0.00000001);
        }

        private SoftmaxLayer softmax;
    }
}
