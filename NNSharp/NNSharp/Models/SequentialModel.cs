using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.Executors;
using NNSharp.LayerDescriptors;
using NNSharp.Layers;

namespace NNSharp.Models
{
    public class SequentialModel : IModel
    {
        public SequentialModel()
        {
            descriptors = new List<ILayerDescriptor>();
        }

        public void Add(ILayerDescriptor descriptor)
        {
            descriptors.Add(descriptor);
        }

        public void Compile(AbstractExecutor compiler)
        {
            compiler.Compile(descriptors);
            compiled = compiler;
        }

        public IData ExecuteNetwork(IData input)
        {
            return compiled.Execute(input);
        }


        private List<ILayerDescriptor> descriptors;
        private AbstractExecutor compiled;

    }
}
