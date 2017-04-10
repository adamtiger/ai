using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Dense2DLayer : Dense2DKernel, ILayer
    {

        public Dense2DLayer(IData weights)
        {
            this.weights = weights as Data2D;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();
            Dimension dimK = this.weights.GetDimension();

            if (dimI.c != dimK.c)
                throw new Exception("Wrong kernel and input sizes: sizes of channels should match." +
                   " Now: dimI: " + dimI.c + " != dimK: " + dimK.c);

            if (dimI.h != dimK.h)
                throw new Exception("Wrong kernel and input sizes: sizes of heights should match." +
                   " Now: dimI: " + dimI.h + " != dimK: " + dimK.h);

            if (dimI.w != dimK.w)
                throw new Exception("Wrong kernel and input sizes: sizes of widths should match." +
                   " Now: dimI: " + dimI.w + " != dimK: " + dimK.w);


            int outputH = 1;
            int outputW = 1;
            int outputC = dimK.b;
            int outputB = dimI.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }
    }
}
