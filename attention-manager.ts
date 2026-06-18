import tf, { dot, Rank, Tensor } from "@tensorflow/tfjs";
export class SimpleAttentionManager {


    calculateAttentionForQuery(query_vector: Tensor<Rank.R1>, sample: tf.Tensor<tf.Rank.R2>) {
        const attention_weights = this.calculateSimilarityScore(query_vector, sample)
        const normalized_weights = this.normalizeSimilarityScores(attention_weights)
        const soft_maxed_weights = this.softMaxNaive(normalized_weights)

    }
    private calculateSimilarityScore(query_vector: Tensor<Rank.R1>, sample: tf.Tensor<tf.Rank.R2>): Tensor<Rank.R1> {
        // First the scalar product of sample * the query.
        // Meaning  [ firstResult of sampleItem1* query, secondResult of sampleItem2* query, ... ] => [ v1 ,v2 ,v3 ,v4 ]
        // this is how much each part of the sample is close to the query ( higher is closer)
        let attention_scores = dot(sample, query_vector) as Tensor<Rank.R1>;
        return attention_scores
    }

    private normalizeSimilarityScores(attention_scores: Tensor<Rank.R1>): Tensor<Rank.R1> {
        // Purpose of normalization is to get values between 0 and 1 to avoid having values that would denature all vectors)
        // Normalisation is eachItem divided by the sum of all
        const sum_all_weights = tf.sum(attention_scores)
        const normalized_weights: Tensor<Rank.R1> = tf.div(attention_scores, sum_all_weights)
        return normalized_weights
    }



    private softMaxNaive(normalized_weights: Tensor<Rank.R1>) {
        const exponential_weights = tf.exp(normalized_weights) // [e(firstWeight), e(secondWeight), ..., ]
        const sum_of_exponential_weights = tf.sum(exponential_weights)
        const naive_softmax = tf.div(exponential_weights, sum_of_exponential_weights)
        console.log("🔍 ~ softMaxNaive ~ attention-manager.ts:32 ~ sum_of_exponential_weights:", naive_softmax.toString());
        return naive_softmax
    }


}