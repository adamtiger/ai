using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.SequentialLayers
{
    public interface ILayer : IKernel
    {
        void SetInput(IData intput);
        IData GetOutput();
    }
}
