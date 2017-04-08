using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Conv2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor, IData weights = null)
        {
            if (descriptor is Convolution2D)
            {
                Convolution2D conv = descriptor as Convolution2D;

                if (weights == null)
                    throw new Exception("Convolution: missing weights!");

                if (weights is Data2D)
                {
                    if (((weights as Data2D).GetDimension().h != conv.KernelHeight) ||
                        ((weights as Data2D).GetDimension().w != conv.KernelWidth) ||
                        ((weights as Data2D).GetDimension().c != conv.KernelChannel) ||
                        ((weights as Data2D).GetDimension().b != conv.KernelNum))
                    {
                        throw new Exception("Convolution: kernel has wrong size!");
                    }
                }

                ILayer layer = new Conv2DLayer(conv.PaddingVertical, conv.PaddingHorizontal,
                                               conv.StrideVertical, conv.StrideHorizontal, weights);

                return layer;
            }

            return null;
        }
    }
}
