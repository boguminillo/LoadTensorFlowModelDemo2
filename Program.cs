using System.Runtime.InteropServices;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

MLContext mlContext = new();

ITransformer model = GetPredictionPipeline(mlContext);

(DataFrame labels, DataFrame data) = GetData();

IDataView inputDataView = generateInput(mlContext, data);

IDataView prediction = model.Transform(inputDataView);

ModelOutput[] predictions = mlContext.Data.CreateEnumerable<ModelOutput>(prediction, reuseRowObject: false).ToArray();

double[] mae = new double[2];
double[] rmse = new double[2];
for (int i = 0; i < predictions.Length; i++)
{
    Console.WriteLine($"Predicted/actual MPG: {predictions[i].MPG}/{labels.Rows[i][0]}\tPredicted/actual acceleration: {predictions[i].Acceleration}/{labels.Rows[i][1]}");
    mae[0] += Math.Abs(predictions[i].MPG - (float)labels.Rows[i][0]);
    mae[1] += Math.Abs(predictions[i].Acceleration - (float)labels.Rows[i][1]);
    rmse[0] += Math.Pow(predictions[i].MPG - (float)labels.Rows[i][0], 2);
    rmse[1] += Math.Pow(predictions[i].Acceleration - (float)labels.Rows[i][1], 2);
}
mae[0] /= predictions.Length;
mae[1] /= predictions.Length;
rmse[0] /= predictions.Length;
rmse[1] /= predictions.Length;
rmse[0] = Math.Sqrt(rmse[0]);
rmse[1] = Math.Sqrt(rmse[1]);
Console.WriteLine($"Mean absolute error for MPG: {mae[0]}\tMean absolute error for acceleration: {mae[1]}");
Console.WriteLine($"Root mean squared error for MPG: {rmse[0]}\tRoot mean squared error for acceleration: {rmse[1]}");

static (DataFrame, DataFrame) GetData()
{
    string[] names = [
        "MPG",
        "Cylinders",
        "Displacement",
        "Horsepower",
        "Weight",
        "Acceleration",
        "Model Year",
        "Origin",
        "Car Name"
        ];

    DataFrame rawData = DataFrame.LoadCsv("C:\\Users\\admnoraybio\\Source\\Repos\\LoadTensorFlowModelDemo2\\data\\auto-mpg.csv",
        header: false,
        columnNames: names,
        dataTypes: [typeof(float), typeof(float), typeof(float), typeof(float), typeof(float), typeof(float), typeof(float), typeof(int), typeof(string)]);

    DataFrame labels = new DataFrame(rawData.Columns["MPG"], rawData.Columns["Acceleration"]);
    rawData.Columns.Remove("MPG");
    rawData.Columns.Remove("Acceleration");

    rawData.Columns.Remove("Car Name");
    
    rawData.Columns.Add(new PrimitiveDataFrameColumn<float>("USA", rawData["Origin"].ElementwiseEquals(1).Select(x => x.HasValue && x.Value ? 1f : 0)));
    rawData.Columns.Add(new PrimitiveDataFrameColumn<float>("Europe", rawData["Origin"].ElementwiseEquals(2).Select(x => x.HasValue && x.Value ? 1f : 0)));
    rawData.Columns.Add(new PrimitiveDataFrameColumn<float>("Japan", rawData["Origin"].ElementwiseEquals(3).Select(x => x.HasValue && x.Value ? 1f : 0)));
    rawData.Columns.Remove("Origin");

    return (labels, rawData);
}

static ITransformer GetPredictionPipeline(MLContext mlContext)
{
    string ONNX_MODEL_PATH = "C:\\Users\\admnoraybio\\Source\\Repos\\LoadTensorFlowModelDemo2\\data\\model.onnx";

    OnnxScoringEstimator pipeline = mlContext
        .Transforms
        .ApplyOnnxModel(
            modelFile: ONNX_MODEL_PATH,
            inputColumnNames: ["input"],
            outputColumnNames: ["output"]
        );

    IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<ModelInput>());

    return pipeline.Fit(dataView);
}

static IDataView generateInput(MLContext mlContext, DataFrame data)
{
    ModelInput[] input = new ModelInput[data.Rows.Count];
    for (int i = 0; i < data.Rows.Count; i++)
    {
        DataFrameRow row = data.Rows[i];
        input[i] = new ModelInput
        {
            Cylinders = (float)row[0],
            Displacement = (float)row[1],
            Horsepower = (float)row[2],
            Weight = (float)row[3],
            ModelYear = (float)row[4],
            USA = (float)row[5],
            Europe = (float)row[6],
            Japan = (float)row[7]
        };
    }
    return mlContext.Data.LoadFromEnumerable(input);
}

public class ModelInput
{
    [ColumnName("input"), VectorType(8)]
    public float[] Input => [this.Cylinders, this.Displacement, this.Horsepower, this.Weight, this.ModelYear, this.USA, this.Europe, this.Japan];

    public float Cylinders { get; set; }

    public float Displacement { get; set; }

    public float Horsepower { get; set; }

    public float Weight { get; set; }

    public float ModelYear { get; set; }

    public float USA { get; set; }

    public float Europe { get; set; }

    public float Japan { get; set; }
}

public class ModelOutput
{
    [ColumnName("output"), VectorType(2)]
    public float[] Score { get; set; }
    
    public float MPG => this.Score[0];

    public float Acceleration => this.Score[1];
}
