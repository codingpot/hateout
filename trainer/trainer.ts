import * as use from '@tensorflow-models/universal-sentence-encoder'
import * as tf from '@tensorflow/tfjs-node'

import path from 'path'
import { loadUnsmileTrainValidData } from './datasets'
import { getModel } from './model'

// model will be saved into ${MODEL_DIRECTORY_PATH}/{model.json,weights.bin}
const MODEL_DIRECTORY_PATH = `file://${path.join(
  __dirname,
  '..',
  'model',
  'ver20220624',
)}`

async function main() {
  const encoder = await use.load()
  const { trainData, valData } = await loadUnsmileTrainValidData(encoder)
  const model = await getModel(MODEL_DIRECTORY_PATH)

  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.sigmoidCrossEntropy,
    metrics: [
      tf.metrics.binaryAccuracy,
      tf.metrics.precision,
      tf.metrics.recall,
    ],
  })

  model.summary()

  await model.fitDataset(trainData, {
    epochs: 1,
    validationData: valData,
    callbacks: [
      tf.callbacks.earlyStopping({
        patience: 1,
      }),
    ],
  })

  const savedResult = await model.save(MODEL_DIRECTORY_PATH)

  if (savedResult.errors) {
    console.error(savedResult)
  } else {
    console.info(savedResult)
  }
}

main()
