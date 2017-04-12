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
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Convolution2D)
            {
                Convolution2D conv = descriptor as Convolution2D;

                Data2D weights = new Data2D(conv.KernelHeight, conv.KernelWidth,
                                            conv.KernelChannel, conv.KernelNum);
                weights.ToZeros();

                ILayer layer = new Conv2DLayer(conv.PaddingVertical, conv.PaddingHorizontal,
                                               conv.StrideVertical, conv.StrideHorizontal, weights);

                return layer;
            }

            return null;
        }
    }
}
