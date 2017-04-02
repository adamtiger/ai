using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.LayerDescriptors;

namespace NNSharp.Layers.DefaultLayers
{
    public class ReLuLayerFactory : IAbstractLayerFactory
    {
        public ILayer CreateProduct(ILayerDescriptor descriptor)
        {
            if (descriptor is ReLu)
                return new ReLuLayer();

            return null;
        }
    }
}
