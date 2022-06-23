import * as use from "@tensorflow-models/universal-sentence-encoder";
import * as tf from "@tensorflow/tfjs-node";

async function main() {
  const encoder = await use.load();
  const train_data = tf.data
    .csv(
      "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_train_v1.0.tsv",
      {
        delimiter: "\t",
        hasHeader: true,
        configuredColumnsOnly: true,
        columnConfigs: {
          clean: {
            dtype: "int32",
            isLabel: true,
          },
          문장: {
            dtype: "string",
          },
        },
      }
    )
    .filter((data: any) => data.xs["clean"] != "")
    .mapAsync(async (data: any) => {
      const out = await encoder.embed(data.xs["문장"]);
      return {
        xs: out.flatten(),
        ys: Object.values(data.ys),
      };
    })
    .batch(32)
    .shuffle(32);

    const val_data = tf.data
    .csv(
      "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_valid_v1.0.tsv",
      {
        delimiter: "\t",
        hasHeader: true,
        configuredColumnsOnly: true,
        columnConfigs: {
          clean: {
            dtype: "int32",
            isLabel: true,
          },
          문장: {
            dtype: "string",
          },
        },
      }
    )
    .filter((data: any) => data.xs["clean"] != "")
    .mapAsync(async (data: any) => {
      const out = await encoder.embed(data.xs["문장"]);
      return {
        xs: out.flatten(),
        ys: Object.values(data.ys),
      };
    })
    .batch(32)
    .shuffle(32);    

  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputDim: 512,
        units: 512,
        activation: "relu",
      }),
      tf.layers.batchNormalization(),
      tf.layers.dense({
        units: 512,
        activation: "relu",
      }),
      tf.layers.batchNormalization(),
      tf.layers.dense({
        units: 1,
        activation: "sigmoid",
      }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.sigmoidCrossEntropy,
    metrics: [tf.metrics.binaryAccuracy],
  });

  model.fitDataset(train_data, {
    epochs: 5,
    validationData: val_data
  });
}

main();
