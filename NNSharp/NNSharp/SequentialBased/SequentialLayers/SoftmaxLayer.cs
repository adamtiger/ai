using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class SoftmaxLayer : SoftmaxKernel, ILayer
    {

        public IData GetOutput()
        {
            return data;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("SoftmaxLayer: input is null.");
            else if (!(input is DataArray))
                throw new Exception("SoftmaxLayer: input is not DataArray.");

            this.data = input as DataArray;

        }
    }
}
