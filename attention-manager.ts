import tf, { dot, Rank, Tensor } from "@tensorflow/tfjs";
import { tensorGet } from "./utils/tensor-get";

export class SimpleAttentionManager {


    calculateAttentionForQuery(query_vector: Tensor<Rank.R1>, sample: tf.Tensor<tf.Rank.R2>) {
        const attention_weights = this.calculateAttentionWeight(query_vector, sample)
        const normalized_weights = this.normalizeAttentionScores(attention_weights)

    }
    private calculateAttentionWeight(query_vector: Tensor<Rank.R1>, sample: tf.Tensor<tf.Rank.R2>): Tensor<Rank.R1> {

        const sample_size = sample.shape[0]
        let attention_scores: Tensor<Rank.R1>

        return attention_scores
    }

    private normalizeAttentionScores(attention_scores: Tensor<Rank.R0>[]): Tensor<Rank.R0>[] {
        const normalized_weights: Tensor<Rank.R0>[] = []
        console.log(attention_scores.length)
        return normalized_weights
    }

    private scalarProduct(query: Tensor<Rank.R1>, external_vector: Tensor<Rank.R1>): Tensor<Rank.R0> {
        return dot(query, external_vector) as Tensor<Rank.R0>;
    }

    private softMax() {
        //@todo
    }


}