using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.Kernels.Kernels;

namespace NNSharp.Kernels.SequentialLayers
{
    public class ReLuLayer : ReLuKernel, ILayer
    {
        
        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            this.input = input; 
        }
    }
}
