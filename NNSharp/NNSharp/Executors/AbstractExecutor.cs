using NNSharp.LayerDescriptors;
using NNSharp.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Compilers
{
    public abstract class AbstractExecutor
    {
        public AbstractExecutor(IAbstractLayerFactory factory)
        {
            this.factory = factory;
        }

        public abstract void Compile(List<ILayerDescriptor> descriptors);

        public abstract IData Execute(IData input);

        protected ILayer CreateLayer(ILayerDescriptor descriptor)
        {
            return factory.CreateProduct(descriptor);
        }

        private IAbstractLayerFactory factory;
    }
}
