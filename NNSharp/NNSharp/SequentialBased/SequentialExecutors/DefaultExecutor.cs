using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.KernelDescriptors;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace NNSharp.SequentialBased.SequentialExecutors
{
    public class DefaultExecutor : ISequentialExecutor
    {
        public DefaultExecutor()
        {
            //this.factory = new ....;
            layers = new List<ILayer>();
        }

        public void Compile(List<IKernelDescriptor> descriptors)
        {
            // The first descriptor shows the size of the input.
            IData initInput = factory.CreateProduct(descriptors[0]).GetOutput();

            // Instantiate the kernels.
            for (int idx = 1; idx < descriptors.Count; ++idx)
            {
                ILayer layer = factory.CreateProduct(descriptors[idx]);
                layers.Add(layer);
            }

            // Propagate through the data sizes and instantiate suitable data types.
            IData input = initInput;
            foreach(var l in layers)
            {
                l.SetInput(input);
                l.Execute();
                input = l.GetOutput();
            }    
        }

        public IData Execute(IData input)
        {
            layers[0].SetInput(input);
            layers.ForEach(l => { l.Execute(); });
            return layers.Last().GetOutput();
        }

        private List<ILayer> layers;
        private IAbstractLayerFactory factory;
    }
}
