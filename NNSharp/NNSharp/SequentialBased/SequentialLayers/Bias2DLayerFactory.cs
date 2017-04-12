using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Bias2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Bias2D)
            {
                Bias2D bias = descriptor as Bias2D;

                DataArray biases = new DataArray(bias.Units);
                biases.ToZeros();

                ILayer layer = new Bias2DLayer(biases);

                return layer;
            }

            return null;
        }
    }
}
