using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.SequentialExecutors;
using NNSharp.LayerDescriptors;
using NNSharp.Kernels;

namespace NNSharp.Models
{
    public class SequentialModel
    {
        public SequentialModel()
        {
            descriptors = new List<ILayerDescriptor>();
        }

        public void Add(ILayerDescriptor descriptor)
        {
            descriptors.Add(descriptor);
        }

        public void Compile(ISequentialExecutor compiler)
        {
            compiler.Compile(descriptors);
            compiled = compiler;
        }

        public IData ExecuteNetwork(IData input)
        {
            return compiled.Execute(input);
        }


        private List<ILayerDescriptor> descriptors;
        private ISequentialExecutor compiled;

    }
}
