import * as use from '@tensorflow-models/universal-sentence-encoder'
import * as tf from '@tensorflow/tfjs-node'

/**
 * 스마일게이트 데이터셋을 universal-sentence-encoder 를 통해 encoding한 tf.data.Dataset을 반환한다.
 * @param filepath 데이터셋 CSV URL @see getUnsmileDataUrl
 * @param encoder use.UniversalSentenceEncoder 를 사용하여 string을 인코딩
 * @link https://github.com/smilegate-ai/korean_unsmile_dataset
 */
async function loadUnsmileData({
  filepath,
  encoder,
}: {
  filepath: string
  encoder: use.UniversalSentenceEncoder
}): Promise<tf.data.Dataset<tf.TensorContainer>> {
  return tf.data
    .csv(filepath, {
      delimiter: '\t',
      hasHeader: true,
      configuredColumnsOnly: true,
      columnConfigs: {
        clean: {
          dtype: 'int32',
          isLabel: true,
        },
        문장: {
          dtype: 'string',
        },
      },
    })
    .mapAsync(async (data: any) => {
      const out = await encoder.embed(data.xs['문장'])
      return {
        xs: out.flatten(),
        ys: Object.values(data.ys),
      }
    })
    .batch(32)
    .shuffle(32)
}

/**
 * 스마일게이트 데이터셋을 universal-sentence-encoder 를 통해 encoding한 tf.data.Dataset을 반환한다.
 * 학습 데이터와 밸리데이션 데이터를 tf.data.Dataset 형태로 반환한다.
 *
 * @param encoder use.UniversalSentenceEncoder
 * @returns
 */
export async function loadUnsmileTrainValidData(
  encoder: use.UniversalSentenceEncoder,
): Promise<{
  trainData: tf.data.Dataset<tf.TensorContainer>
  valData: tf.data.Dataset<tf.TensorContainer>
}> {
  const trainData = await loadUnsmileData({
    filepath: getUnsmileDataUrl('train', 'v1.0'),
    encoder,
  })
  const valData = await loadUnsmileData({
    filepath: getUnsmileDataUrl('valid', 'v1.0'),
    encoder,
  })
  return { trainData, valData }
}

/**
 * 스마일게이트 데이터셋 CSV URL을 위한 도움 함수.
 *
 * @param type "train" or "valid"
 * @param version "v1.0"
 * @returns full url path
 */
function getUnsmileDataUrl(type: string, version: string): string {
  return `https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_${type}_${version}.tsv`
}
