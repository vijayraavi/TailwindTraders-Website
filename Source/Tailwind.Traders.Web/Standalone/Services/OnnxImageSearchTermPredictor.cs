using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Transforms.Onnx;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;

namespace Tailwind.Traders.Web.Standalone.Services
{
    public class OnnxImageSearchTermPredictor : IImageSearchTermPredictor
    {
        private readonly ILogger<OnnxImageSearchTermPredictor> logger;
        private readonly InferenceSession session;

        public OnnxImageSearchTermPredictor(IHostingEnvironment environment, ILogger<OnnxImageSearchTermPredictor> logger)
        {
            this.logger = logger;
            logger.LogInformation("ctor");
            var file = System.IO.File.ReadAllBytes(Path.Combine(environment.ContentRootPath, "Standalone/OnnxModels/products.onnx"));
            //session = new InferenceSession(Path.Combine(environment.ContentRootPath, "Standalone/OnnxModels/products.onnx"));
        }

        public Task<string> PredictSearchTerm(Stream imageStream)
        {
            DenseTensor<float> data = ConvertImageToTensor(imageStream);
            var input = NamedOnnxValue.CreateFromTensor<float>("data", data);
            using (var output = session.Run(new[] { input }))
            {
                var prediction = output.First(i => i.Name == "classLabel").AsEnumerable<string>().First();
                return Task.FromResult(prediction);
            }
        }

        private DenseTensor<float> ConvertImageToTensor(Stream imageStream)
        {
            logger.LogInformation("ConvertImageToTensor");
            var data = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            using (var image = Image.Load(imageStream))
            {
                image.Mutate(ctx => ctx.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Stretch
                }));
                for (var x = 0; x < image.Width; x++)
                {
                    for (var y = 0; y < image.Height; y++)
                    {
                        var color = image.GetPixelRowSpan(y)[x];
                        data[0, 0, x, y] = color.B;
                        data[0, 1, x, y] = color.G;
                        data[0, 2, x, y] = color.R;
                    }
                }
            }
            return data;
        }

        private PredictionEngine<ImageInput, ImagePrediction> LoadModel(string onnxModelFilePath)
        {
            var ctx = new MLContext();
            var dataView = ctx.Data.LoadFromEnumerable(new List<ImageInput>());
            var pipeline = ctx.Transforms.ApplyOnnxModel(
                modelFile: onnxModelFilePath, 
                outputColumnNames: new[] { "classLabel", "loss" }, inputColumnNames: new[] { "data" });

            var model = pipeline.Fit(dataView);
            return ctx.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(model);
        }
    }

    public class ImagePrediction
    {
        [ColumnName("classLabel")]
        [VectorType]
        public string[] Prediction;

        [OnnxSequenceType(typeof(IDictionary<string, float>))]
        public IEnumerable<IDictionary<string, float>> loss;
    }

    public class ImageInput
    {
        [VectorType(3, 224, 224)]
        [ColumnName("data")]
        public float[] Data { get; set; }
    }
}