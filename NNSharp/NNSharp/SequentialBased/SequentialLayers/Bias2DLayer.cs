using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Bias2DLayer : Bias2DKernel, ILayer
    {

        public Bias2DLayer(IData biases)
        {
            this.biases = biases as DataArray;
        }
 
        public IData GetOutput()
        {
            return input;
        }

        public void SetInput(IData input)
        {
            this.input = input as Data2D;

            int a, b;
            if ((a = this.input.GetDimension().b) == (b =this.biases.GetLength()))
                throw new Exception("Bias: the number of biases is not suitable -> "+ a + " != " + b);
        }
    }
}
