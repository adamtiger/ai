using NNSharp.Kernels;
using NNSharp.LayerDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialExecutors
{
    public interface ISequentialExecutor
    {
        void Compile(List<ILayerDescriptor> descriptors);
        IData Execute(IData input);
    }
}
