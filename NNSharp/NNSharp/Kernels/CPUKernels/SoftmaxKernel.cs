using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    public class SoftmaxKernel : IKernel
    {
        public void Execute()
        {
            double sum = 0.0;
            foreach (var x in data)
            {
                sum += Math.Exp(x);
            }

            data.ApplyToAll(x => { return Math.Exp(x) / sum; });      
        }

        protected DataArray data;
    }
}
