using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.DataTypes
{

    public class Data2D : IData, IEnumerable<double>
    {

        public Data2D(int height, int width, int channels = 3, int batchSize = 1)
        {
            tensor = new double[height, width, channels, batchSize];
            D = new Dimension(height, width, channels, batchSize);
        }


        public double this[int h, int w, int c, int b]
        {
            get
            {
                return tensor[h, w, c, b];
            }

            set
            {
                tensor[h, w, c, b] = value;
            }
        }

        public IEnumerator<double> GetEnumerator()
        {
            return (IEnumerator<double>)tensor.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return tensor.GetEnumerator();
        }

        public void ApplyToAll(Operation operation)
        {
            for (int b = 0; b < D.b; ++b)
            {
                for (int c = 0; c < D.c; ++c)
                {
                    for (int w = 0; w < D.w; ++w)
                    {
                        for (int h = 0; h < D.h; ++h)
                        {
                            tensor[h, w, c, b] = operation(tensor[h, w, c, b]);
                        }
                    }
                }
            }
        }

        public void ToZeros()
        {
            this.ApplyToAll(p => { return 0.0; });
        }

        public Dimension GetDimension()
        {
            return D;
        }

        private double[,,,] tensor;

        public struct Dimension
        {
            public Dimension(int h, int w, int c, int b)
            {
                this.h = h; this.w = w;
                this.c = c; this.b = b;
            }
            public int h; // height
            public int w;
            public int c;
            public int b;
        } private Dimension D;

    }
}
