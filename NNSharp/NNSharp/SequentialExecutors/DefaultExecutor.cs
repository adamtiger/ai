using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.LayerDescriptors;
using NNSharp.Kernels;
using NNSharp.Kernels.SequentialLayers;

namespace NNSharp.SequentialExecutors
{
    public class DefaultExecutor : ISequentialExecutor
    {
        public DefaultExecutor(IAbstractLayerFactory factory)
        {

        }

        public void Compile(List<ILayerDescriptor> descriptors)
        {
            throw new NotImplementedException();
        }

        public IData Execute(IData input)
        {
            throw new NotImplementedException();
        }
    }
}
