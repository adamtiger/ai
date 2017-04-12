using Newtonsoft.Json.Linq;
using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            JObject model = JObject.Parse(File.ReadAllText("test.json"));

            String modelType = (String)model.SelectToken("model_type");

            List<IKernelDescriptor> dscps = model.SelectToken("descriptors").Select(layer => {

                IKernelDescriptor descriptor = null;

                String layerName = (String)layer.SelectToken("layer");

                switch (layerName)
                {
                    case "Convolution2D":

                        break;
                }

                return descriptor;
            }).ToList();

            Console.WriteLine(modelType);


        }
    }
}
