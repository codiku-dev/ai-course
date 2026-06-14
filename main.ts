import { Tokenizer } from "./tokenizer";
import { DataSetV1 } from "./dataset-v1";
import { DataSetV2 } from "./dataset-v2";
import fs from "fs";
import * as tf from "@tensorflow/tfjs";
// import { DataLoader } from "./data-loader";
import { DataLoader as DataLoaderV2 } from "./data-loader-v2";
import { DataSetV3 } from "./dataset-v3";
import { EmbeddingManager } from "./embedding-manager";
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

const EMBEDDINGS_DIMENSIONS = 256;

const embedding_manager = new EmbeddingManager({ dimensions: EMBEDDINGS_DIMENSIONS });
// embedding_manager.initializeEmbeddings({ 
// vocabulary_file_path: "./vocabulary.json", 
// output_file_path: "./embeddings.json", 
// dimensions: 256 });
embedding_manager.loadExistingEmbeddings("./embeddings.json");


const data_set_v3 = new DataSetV3(
  {
    filePath: "data/the-verdict.txt",
    tokenizer: tokenizer,
    with_logger: false
  },
);

const BATCH_WIDTH = 4
const BATCH_HEIGHT = 8
const data_loader_v2 = new DataLoaderV2({
  dataset: data_set_v3,
  batch_size: BATCH_HEIGHT,
  shuffle: false,
  numberOfItemsPerRow: BATCH_WIDTH,
  stride: 1,
});

const first_batch = data_loader_v2.next();
data_loader_v2.logLastBatch()
// console.log(first_batch.inputSamples.shape)
// const secondBatch = dataLoader.next();
// dataLoader.logLastBatch();

const embeddings = embedding_manager.getEmbeddings();
// console.log(embeddings.shape)
const batch_input_embeddings = tf.gather(embeddings, first_batch.inputSamples, 0);
console.log(batch_input_embeddings.shape)
// [0, 1, 2, 3]

const position_scalars = tf.add(
  tf.zeros([BATCH_HEIGHT, BATCH_WIDTH], "float32"),
  tf.range(0, BATCH_WIDTH, 1, "float32")
);
console.log(position_scalars.shape)
console.log(position_scalars.toString())

/*

[ 8, 4 ]
Tensor
    [[0, 1, 2, 3],
     [0, 1, 2, 3],
     [0, 1, 2, 3],
     [0, 1, 2, 3],
     [0, 1, 2, 3],
     [0, 1, 2, 3],
     [0, 1, 2, 3],
     [0, 1, 2, 3]]

     */



const positional_embeddings_to_apply = position_scalars
  .expandDims(-1)     // Ajoute une dimension à [8, 4] pour avoir [8, 4, 1]    (-1 signifie la dernière dimension a modifier)  
  // 
  /*[[[0], [1], [2], [3]],
  [[0], [1], [2], [3]],
    ...
  [[0], [1], [2], [3]],
]
  */                  // [8, 4, 1]
  .tile([1, 1, EMBEDDINGS_DIMENSIONS]);

/*

.tile([1, 1, 256])
     ↑  ↑   ↑
     │  │   └── dim 2 : répéter 256 fois  →  [1] devient [0,0,0,...,0]
     │  └────── dim 1 : répéter 1 fois   →  inchangé (4 cols)
     └───────── dim 0 : répéter 1 fois   →  inchangé (8 lignes)
     */

/* On a au final :
[8, 4, 256]
*/

const embeddings_with_positional_embeddings = tf.add(batch_input_embeddings, positional_embeddings_to_apply);
console.log(embeddings_with_positional_embeddings.shape)

console.log("initial input embeddings")
console.log(batch_input_embeddings.toString())
console.log("positional embeddings")
console.log(positional_embeddings_to_apply.toString())
console.log("embeddings with positional embeddings")
console.log(embeddings_with_positional_embeddings.toString())

/*
[8, 4, 512]
*/
