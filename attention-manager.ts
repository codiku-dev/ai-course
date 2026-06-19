import tf, { dot, matMul, Rank, Tensor } from "@tensorflow/tfjs";
export class SimpleAttentionManager {

  calculateAttentionForSample(
    sample: tf.Tensor<tf.Rank.R2>
  ) {
    // il faudrait boucler sur les samples et créer une matrice d'attention pour ce sample.

    // ON fera une boucle for et pour chaque item du sample ca fera une query qui retournerai  calculateAttentionForQuery()
    // on aurai autant de vecteur de context que d'item dnas le sample
    //  Probleme ca ne tourne pas le sur gpu
    // il faut donc tout faire en tensor
    const similarity_weights_sample = this.calculateSimilarityScoreForSample(sample)
    console.log("similarity_weights_sample", similarity_weights_sample.toString());
    const softmax_sample = this.applySoftMaxToSample(similarity_weights_sample)
    console.log("softmax_sample", softmax_sample.toString())
    return similarity_weights_sample;
  }



  calculateAttentionForQuery(
    query_vector: Tensor<Rank.R1>,
    sample: tf.Tensor<tf.Rank.R2>,
  ) {
    let similarities_weights = this.calculateSimilarityScoreForQuery(
      query_vector,
      sample,
    );
    console.log("similarities_weights", similarities_weights.toString());
    const softmaxed_weights = this.applySoftMax(similarities_weights);
    console.log("normalized_weights", softmaxed_weights.toString());
    console.log("Is sum of divisions 1 ?", tf.sum(softmaxed_weights).toString());

    const context_vector = this.calculateContextVector(sample, softmaxed_weights);
    console.log("context_vector", context_vector.toString());
    return context_vector;
  }

  private calculateSimilarityScoreForSample(sample: tf.Tensor<tf.Rank.R2>): Tensor<Rank.R2> {

    //  c'est pas possible mathématique car il faut width A === height B donc on transpose B
    // et meme si la matrice est carré, On transpose B, sinon on aurait multiplié token1 par des bout du token1, bout de token 2,3 .. puis token2 par des bout du 1,2 ,3
    // ce qui ne fait aucun sens
    let attention_scores_sample = matMul(sample, sample, false, true) as Tensor<Rank.R2>;
    return attention_scores_sample

  }
  private calculateSimilarityScoreForQuery(
    query_vector: Tensor<Rank.R1>,
    sample: tf.Tensor<tf.Rank.R2>,
  ): Tensor<Rank.R1> {
    // First the scalar product of sample * the query.
    // Meaning  [ firstResult of sampleItem1* query, secondResult of sampleItem2* query, ... ] => [ v1 ,v2 ,v3 ,v4 ]
    // this is how much each part of the sample is close to the query ( higher is closer)
    let attention_scores = dot(sample, query_vector) as Tensor<Rank.R1>;
    return attention_scores;
  }

  private applySoftMax(attention_weights: Tensor<Rank.R1>): Tensor<Rank.R1> {
    // Formula pytorch should be something like  [ (e(x1-max)) / sum , (e(x2-max)) / sum ,...,  (e(xn-max)) / sum  ]
    const max_weight = tf.max(attention_weights) as Tensor<Rank.R0>;

    const numerators = tf.exp(tf.sub(attention_weights, max_weight)) as Tensor<Rank.R1>;
    // const numerators = tf.exp(tf.sub(attention_weights, max_weight))
    const denominators = tf.sum(numerators) as Tensor<Rank.R1>;
    const divisions = tf.div(numerators, denominators) as Tensor<Rank.R1>;

    return divisions;
  }

  private applySoftMaxToSample(attention_weights: Tensor<Rank.R2>): Tensor<Rank.R2> {
    // Formula pytorch should be something like  [ (e(x1-max)) / sum , (e(x2-max)) / sum ,...,  (e(xn-max)) / sum  ]
    // Le true ici evite que ca devienne un vecteur a 1 seule dimension mais qu'on garde les 2 pour apres
    const max_weight = tf.max(attention_weights, 1, true) as Tensor<Rank.R2>;
    console.log("max_weight", max_weight.toString());
    console.log("max_weigth shape", max_weight.shape)

    const weight_minus_max = tf.sub(attention_weights, max_weight)
    console.log("weight_minus_max", weight_minus_max.toString())
    // ici on garde la meme shape , on a  une liste de numerator et chaque ligne correspond a un token
    const numerators = tf.exp(weight_minus_max) as Tensor<Rank.R2>;
    console.log("numerators", numerators.toString())
    // On some les numerators sur l'axe X de la matrice, donc on obtenir aussi le denominateur pour chaque ligne ( token )
    // On précise que l'addition se fait sur chaque ligne (x)
    const denominators = tf.sum(numerators, 1, true) as Tensor<Rank.R2>;
    // on divise tout
    const divisions = tf.div(numerators, denominators) as Tensor<Rank.R2>;

    return divisions;
  }


  private calculateContextVector(sample: tf.Tensor<tf.Rank.R2>, attention_weights: Tensor<Rank.R1>) {
    // each token is multiple by it's associated attention weight a
    return dot(attention_weights, sample) as Tensor<Rank.R2>;
  }
}
