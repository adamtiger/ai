using NNSharp.Kernels.SequentialLayers;
using NNSharp.LayerDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.SequentialLayers
{
    public interface IAbstractLayerFactory
    {
        ILayer CreateProduct(ILayerDescriptor descriptor);
    }
}
