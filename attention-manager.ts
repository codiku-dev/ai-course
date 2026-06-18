import tf, { dot, Rank, Tensor } from "@tensorflow/tfjs";
export class SimpleAttentionManager {
  calculateAttentionForQuery(
    query_vector: Tensor<Rank.R1>,
    sample: tf.Tensor<tf.Rank.R2>,
  ) {
    let similarities_weights = this.calculateSimilarityScore(
      query_vector,
      sample,
    );
    console.log("similarities_weights", similarities_weights.toString());
    const softmaxed_weights = this.applySoftMax(similarities_weights);
    console.log("normalized_weights", softmaxed_weights.toString());
    return softmaxed_weights;
  }

  private calculateSimilarityScore(
    query_vector: Tensor<Rank.R1>,
    sample: tf.Tensor<tf.Rank.R2>,
  ): Tensor<Rank.R1> {
    // First the scalar product of sample * the query.
    // Meaning  [ firstResult of sampleItem1* query, secondResult of sampleItem2* query, ... ] => [ v1 ,v2 ,v3 ,v4 ]
    // this is how much each part of the sample is close to the query ( higher is closer)
    let attention_scores = dot(sample, query_vector) as Tensor<Rank.R1>;
    return attention_scores;
  }

  private applySoftMax(attention_weights: Tensor<Rank.R1>) {
    // Formula pytorch should be something like  [ (e(x1-max)) / sum , (e(x2-max)) / sum ,...,  (e(xn-max)) / sum  ]
    const max_weight = tf.max(attention_weights)
    // naive version that does not remove max
    const numerators = tf.exp(tf.sub(attention_weights, max_weight))
    // const numerators = tf.exp(tf.sub(attention_weights, max_weight))
    const denominators = tf.sum(numerators)
    return tf.div(numerators, denominators)
  }
}
