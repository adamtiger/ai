using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.LayerDescriptors;
using NNSharp.Layers;

namespace NNSharp.Executors
{
    public class DefaultExecutor : AbstractExecutor
    {
        public DefaultExecutor(IAbstractLayerFactory factory) : base(factory)
        {

        }

        public override void Compile(List<ILayerDescriptor> descriptors)
        {
            throw new NotImplementedException();
        }

        public override IData Execute(IData input)
        {
            throw new NotImplementedException();
        }
    }
}
