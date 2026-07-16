import tf, { Tensor, Rank } from "@tensorflow/tfjs";
type Props = {
    dim_input: number;
    dim_output: number;
    context_length: number;
}
export class CausalAttentionManager {
    private w_q: Tensor<Rank.R2>;
    private w_k: Tensor<Rank.R2>;
    private w_v: Tensor<Rank.R2>;
    private causal_mask_tensor: Tensor<Rank.R2>;
    constructor({ dim_input, dim_output, context_length }: Props) {
        this.w_k = tf.randomNormal(([dim_input, dim_output]))
        this.w_q = tf.randomNormal(([dim_input, dim_output]))
        this.w_v = tf.randomNormal(([dim_input, dim_output]))
        //Fixed so calculated only once ( natively move to gpu  when part of a calculus with gpu elements)
        this.causal_mask_tensor = tf.linalg.bandPart(tf.ones([context_length, context_length]), -1, 0);

    }
    /* Apply a causal attention with mask and dropout and finally return the context vectors */
    forward(batch_input: Tensor<Rank.R3>) {
        const keys = tf.matMul(batch_input, this.w_k)
        const queries = tf.matMul(batch_input, this.w_q)
        const values = tf.matMul(batch_input, this.w_v)
        // on transpose que les deux derniers axe
        const keys_transposed = tf.transpose(keys, [0, 2, 1])
        const attention_score = tf.matMul(queries, keys_transposed)
        const last_dim_keys = keys.shape[keys.shape.length - 1]
        const scaled_scores = attention_score.div(Math.sqrt(last_dim_keys))

        // arg -1 is how much to keep pristine below diag , which means all.
        // arg 0 is keep exactly 0 , to replace them by default with 0 ( it's built in to be 0)
        // numLower=-1: keep all below diagonal; numUpper=0: zero everything above
        const causal_mask = this.causal_mask_tensor;
        const masked_scores = tf.where( // Kind of ternary operator
            causal_mask.equal(1), // if cause mask is 1 ?
            scaled_scores, // true => scaled_scores
            Number.NEGATIVE_INFINITY // false => negative infinity
        );

        // On donne la dimension dans laquelle se trouve les valeur a normaliser et softmax var normaliser sur l'ensemble des valeurs
        const attention_weights = tf.softmax(masked_scores, -1);
        /* 50% chance to keep the attention weights.
        Why ? Because we want to avoid overfitting and to have a more robust model.
        What is overfitting ? It's when the model learns the training data too well, and it's not able to generalize to new data.
        Example if you recognize Paul only because he has a beard, you will not be able to recognize him if he doesn't have a beard.
        So it's better to train yourself to recognize paul sometimes thanks to his glasses or his hair.
        */
        const attention_weights_dropout = tf.dropout(attention_weights, 0.4)
        const context_vectors = tf.matMul(attention_weights_dropout, values)
        return context_vectors
    }
}