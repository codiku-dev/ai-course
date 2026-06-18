import { DataSetV1 } from "./dataset-v1";
import { DataSetV2 } from "./dataset-v2";
import fs from "fs";
import * as tf from "@tensorflow/tfjs";
// import { DataLoader } from "./data-loader";
import { DataLoader as DataLoaderV2 } from "./data-loader-v2";
import { DataSetV3 } from "./dataset-v3";
import { EmbeddingManager } from "./embedding-manager";
import { SimpleAttentionManager } from "./attention-manager";
import { Tokenizer } from "./tokenizer";

// const tokenizer = new Tokenizer({
//   with_logger: false,
//   target_vocab_size: 4000,
//   min_pair_frequency: 5,
//   max_token_length: 20,
// });
// Generate vocabulary from corpus
// tokenizer.loadCorpusAsArray("data/corpus-13mb.txt");
// tokenizer.generateVocabulary();
// tokenizer.loadData({
//   path_vocabulary: "vocabulary.json",
//   path_merge: "merge-pairs-js.json",
// });
// tokenizer.decode([1635, 15])
// tokenizer.encode("hello")

// import { get_encoding } from "tiktoken";
// const tokenizerGpt2 = get_encoding("gpt2");
// const tokens = tokenizerGpt2.encode("hello world")
// tokenizerGpt2.free();
// console.log(tokens)

// Data loader and sliding windows for unsupervised training
// First we tokenize the corpus
// On découpe en gardant l'espace collé au mot (split après chaque espace)
// => rapide (BPE par mot) ET les espaces sont conservés pour le round-trip
// const trainingStringArray = fs
//   .readFileSync("data/the-verdict.txt", "utf8")
//   .split(/(?<= )/);
// const trainingTokensArray = [];
// for (const word of trainingStringArray) {
//   trainingTokensArray.push(tokenizer.encode(word));
// }
// const numberOfPredictionTaskPerRow = 4;
// const stride = 4;
//const tensorInput = [];
//const tensorOutput = [];

// Approche avec mots et sans tensors juste avec tableau de input et target ( plusieurs prédition par ligne)

// for (let i = 0; i < trainingStringArray.length; i += numberOfPredictionTaskPerRow) {
//     const inputRow = trainingStringArray.slice(i, i + numberOfPredictionTaskPerRow)
//     const outputRow = trainingStringArray.slice(i + stride, i + stride + numberOfPredictionTaskPerRow)
//     tensorInput.push(inputRow)
//     tensorOutput.push(outputRow)
// }

// Approche avec des vrai token id
// for (let i = 0; i < trainingTokensArray.length; i += numberOfPredictionTaskPerRow) {
//     const inputRow = trainingTokensArray.slice(i, i + numberOfPredictionTaskPerRow)
//     const outputRow = trainingTokensArray.slice(i + stride, i + stride + numberOfPredictionTaskPerRow)
//     tensorInput.push(inputRow)
//     tensorOutput.push(outputRow)
// }

// console.table(tensorInput)
// console.log("\n\n\n")
// console.table(tensorOutput)

// Creation du dataset a la place

// console.log("Params : numberOfPredictionTaskPerRow = ", numberOfPredictionTaskPerRow, "stride = ", stride)
// const dataset = new DataSetV1("data/the-verdict-small.txt", tokenizer, numberOfPredictionTaskPerRow, stride);
// console.table(dataset.toStringInputTensor())
// console.log("\n\n\n")
// console.table(dataset.toStringOutputTensor())

const EMBEDDINGS_DIMENSIONS = 6;
const BATCH_WIDTH = 4;

const tokenizer = new Tokenizer({
  with_logger: false,
});
// Generate vocabulary from corpus
// tokenizer.loadCorpusAsArray("data/corpus-13mb.txt");
// tokenizer.generateVocabulary({ target_vocab_size: 4000, min_pair_frequency: 5, max_token_length: 20 });
tokenizer.loadData({
  path_vocabulary: "vocabulary.json",
  path_merge: "merge-pairs-js.json",
});

const embedding_manager = new EmbeddingManager({
  dimensions: EMBEDDINGS_DIMENSIONS,
  batch_width: BATCH_WIDTH,
});
// embedding_manager.initializeEmbeddings({
//   vocabulary_file_path: "./vocabulary.json",
//   output_file_path: `./embeddings-${EMBEDDINGS_DIMENSIONS}.json`,
//   dimensions: EMBEDDINGS_DIMENSIONS
// });
embedding_manager.loadExistingEmbeddings(`./embeddings-${EMBEDDINGS_DIMENSIONS}.json`);

const data_set_v3 = new DataSetV3({
  filePath: "data/the-verdict.txt",
  tokenizer: tokenizer,
  with_logger: false,
});

const BATCH_HEIGHT = 8;
const data_loader_v2 = new DataLoaderV2({
  dataset: data_set_v3,
  batch_size: BATCH_HEIGHT,
  shuffle: false,
  context_size: BATCH_WIDTH,
  stride: 1,
});

const first_batch = data_loader_v2.next();
data_loader_v2.logLastBatch();
// console.log(first_batch.inputSamples.shape)
// const secondBatch = dataLoader.next();
// dataLoader.logLastBatch();

// const embeddings = embedding_manager.getEmbeddings();
// console.log(embeddings.shape)
// const batch_input_embeddings = tf.gather(embeddings, first_batch.input_samples, 0);
// console.log(batch_input_embeddings.shape)
// [0, 1, 2, 3]

// On fait des vecteurs aléatoires entre -1 et 1 pour les positions ( pour éviter de tout tirer vers le haut en choisissant entre 0 et 1)
// const position_vectors = tf.randomUniform([BATCH_WIDTH, EMBEDDINGS_DIMENSIONS], -1, 1)
/*
Tensor
    [[0.6124023, 0.4720555, 0.7288045, ..., 0.5850454, 0.8543764, 0.3275248],
     [0.7784748, 0.9434558, 0.64477  , ..., 0.7493787, 0.9160827, 0.396835 ],
     [0.7745323, 0.2990477, 0.6029897, ..., 0.3176321, 0.4668534, 0.3826337],
     [0.7151411, 0.7744815, 0.6775624, ..., 0.1684889, 0.156136 , 0.0022091]]

*/

/*
            256 colonnes
          ┌───────────────────────────┐
Colonne 0 → [0.49, 0.12, 0.88, ..., 0.34]   ← position 0 , Colonne 1[0.07, 0.91, 0.23, ..., 0.66]   ← position 1, Colonne 2 → [0.55, 0.40, 0.71, ..., 0.18]   ← position 2, Colonne 3 → [0.81, 0.29, 0.06, ..., 0.93]   ← position 3

          └───────────────────────────

*/

// console.log(position_vectors.shape)
// console.log(position_vectors.toString())

// const input_batch_with_positional_embeddings = tf.add(batch_input_embeddings, position_vectors)
// console.log("batch before positional embeddings")
// console.log(batch_input_embeddings.toString())
// console.log("positional embeddings")
// console.log(position_vectors.toString())
// console.log("batch after positional embeddings")
// console.log(input_batch_with_positional_embeddings.toString())

// Add position vectors to batch input
const batch_with_position_vectors = embedding_manager.forward(first_batch);

const first_sample_with_position_vectors =
  batch_with_position_vectors.input_embeddings
    .slice([0, 0, 0], [1, -1, -1])
    .squeeze() as tf.Tensor<tf.Rank.R2>;

const first_sample_query_vector = first_sample_with_position_vectors
  .slice([0, 0], [1, -1])
  .squeeze() as tf.Tensor<tf.Rank.R1>;

console.log(
  "first_sample_with_position_vectors shape",
  first_sample_with_position_vectors.shape,
);
console.log("first_sample_query_vector shape", first_sample_query_vector.shape);
// console.log("first_sample_query_vector shape", first_sample_query_vector.shape)
// console.log(first_sample_query_vector.toString())
// const queryAsTokenId = first_batch.input_samples.slice([0, 0], [1, 1]).squeeze();
// console.log("Query as token id")
// console.log(queryAsTokenId.toString())
// console.log("Query as token")
// console.log(tokenizer.decode([queryAsTokenId.dataSync()[0]]))
const attention_manager = new SimpleAttentionManager();
//todo attention manager calculate attention for query
const attention_scores_for_query = attention_manager.calculateAttentionForQuery(
  first_sample_query_vector,
  first_sample_with_position_vectors,
);
// console.log(attention_scores_for_query.length)

/*
[8, 4, 256]
embeddings[ligne][colonne][caseDuVecteur]
          ↑        ↑            ↑
       batch    token       dim 256

*/
