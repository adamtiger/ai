using Newtonsoft.Json.Linq;
using NNSharp.DataTypes;
using NNSharp.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.IO
{
    public class ReaderKerasModel
    {
        public ReaderKerasModel(string fname)
        {
            JObject model = JObject.Parse(File.ReadAllText("test.json"));
            String modelType = (String)model.SelectToken("model_type");

            if (!modelType.Equals("Sequential"))
                throw new Exception("This reader only supports Sequential type models!");

            SequentialModel seq = new SequentialModel();


        }

        public SequentialModel GetSequentialExecutor()
        {
            return null;
        }

        private List<IData> weights;
    }
}
