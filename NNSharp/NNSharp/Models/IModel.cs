using NNSharp.Compilers;
using NNSharp.LayerDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Models
{
    public interface IModel
    {
        void Add(ILayerDescriptor descriptor);

        void Compile(AbstractCompiler compiler);

        // Further functions for get the description of the neural network.
    }
}
