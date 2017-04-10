using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Dense2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor, IData weights = null)
        {
            if (descriptor is Dense2D)
            {
                Dense2D dens = descriptor as Dense2D;

                if (weights == null)
                    throw new Exception("Convolution: missing weights!");

                if (weights is Data2D)
                {
                    if ((weights as Data2D).GetDimension().c != dens.Units)
                    {
                        throw new Exception("Dense: kernel has wrong size," + 
                            "the number of output units should math with the batch of the kernel!");
                    }
                }

                ILayer layer = new Dense2DLayer(weights);

                return layer;
            }

            return null;
        }
    }
}
