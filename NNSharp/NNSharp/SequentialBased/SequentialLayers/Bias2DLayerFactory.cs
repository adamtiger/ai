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
        public ILayer CreateProduct(IKernelDescriptor descriptor, IData weights = null)
        {
            if (descriptor is Bias2D)
            {
                Bias2D bias = descriptor as Bias2D;

                if (weights == null)
                    throw new Exception("Bias: missing weights!");

                if (weights is DataArray)
                {
                    if ((weights as DataArray).GetLength() != bias.Units)
                    {
                        throw new Exception("Bias: wrong size.");
                    }
                }
                else
                    throw new Exception("Data type is not DataArray.");

                ILayer layer = new Bias2DLayer(weights);

                return layer;
            }

            return null;
        }
    }
}
