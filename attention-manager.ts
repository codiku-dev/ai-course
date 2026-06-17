import tf, { Rank, Tensor } from "@tensorflow/tfjs";

export class AttentionManager {
    constructor() { // [ [vector256], [vector256], [vector256], [vector256] ... ]
    }

    calculateAttentionForQuery(query: Tensor<Rank.R2>, input_embeddings: Tensor<Rank.R2>) {
        const sample_size = input_embeddings.shape[0]
        for (let i = 0; i < sample_size; i++) {
            // Scalar product between the token vector and the query
            const input_embedding = input_embeddings
                .slice(i)
            console.log("Emedding to compare")
        }
    }


}