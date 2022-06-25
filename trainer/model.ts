import * as tf from '@tensorflow/tfjs-node'
import path from 'path'

const FILE_SCHEME = 'file://'

/**
 * 모델을 불러오거나 불러오는데 실패할 경우 새로운 모델을 생성한다.
 *
 * @param modelDirectoryPath 저장된 모델의 경로. 인풋 형식은 https://www.tensorflow.org/js/guide/save_load 참조 할 것.
 * @returns 학습 모델을 반환
 */
export async function getModel(
  modelDirectoryPath: string,
): Promise<tf.LayersModel | tf.Sequential> {
  try {
    const modelPath =
      FILE_SCHEME +
      path.join(modelDirectoryPath.replace(FILE_SCHEME, ''), 'model.json')
    console.info(`Trying to load a model from ${modelPath}`)
    return await tf.loadLayersModel(modelPath)
  } catch (e) {
    console.warn(e)
    console.warn(`Unable to load a model. Creating a new model`)
    return tf.sequential({
      layers: [
        tf.layers.dense({
          inputDim: 512,
          units: 32,
          activation: 'relu',
        }),
        tf.layers.batchNormalization(),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
        }),
        tf.layers.batchNormalization(),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid',
        }),
      ],
    })
  }
}
