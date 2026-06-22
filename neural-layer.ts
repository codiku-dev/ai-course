import * as tf from "@tensorflow/tfjs"
type NeuralLayerProps = {
    readonly token_vector_dimensions: number,
    readonly vocabulary_size: number,
}
export class NeuralNetwork {
    private neural_network: tf.Tensor<tf.Rank.R2>[] = []
    constructor(config: NeuralLayerProps) {
        const neural_layer: tf.Tensor<tf.Rank.R2> = tf.randomUniform([config.token_vector_dimensions, config.vocabulary_size])
        this.neural_network.push(neural_layer)
    }


}