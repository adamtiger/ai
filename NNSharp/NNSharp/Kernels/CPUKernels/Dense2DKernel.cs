using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class Dense2DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dimK = weights.GetDimension();
            Dimension dimI = input.GetDimension();

            for (int b = 0; b < dimI.b; ++b)
            {
                for (int kernel = 0; kernel < dimK.b; ++kernel)
                {
                    output[0, 0, kernel, b] = 0;
                    for (int c = 0; c < dimI.c; ++c)
                    {
                        for (int h = 0; h < dimI.h; ++h)
                        {
                            for(int w = 0; w < dimI.w; ++w)
                            {
                                output[0, 0, kernel, b] += input[h, w, c, b] * weights[h, w, c, kernel];
                            }
                        }
                    }
                }
            }

        }

        protected Data2D input;
        protected Data2D output;
        protected Data2D weights;
    }
    
}
