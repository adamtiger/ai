using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.IO
{
    public class ReaderKerasWeights
    {
        public ReaderKerasWeights(string fname)
        {

        }

        public IData GetWeightsFor(int idx)
        {
            return weights[idx];
        }

        private List<IData> weights;
    }
}
