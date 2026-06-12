import { Tokenizer } from "./tokenizer";
import { DataSetV1 } from "./dataset-v1";
import { DataSetV2 } from "./dataset-v2";
import fs from "fs";
// import { DataLoader } from "./data-loader";
import { DataLoader as DataLoaderV2 } from "./data-loader-v2";
import { DataSetV3 } from "./dataset-v3";
import { EmbeddingManager } from "./embedding-manager";
const tokenizer = new Tokenizer({
  withLogger: false,
  targetVocabSize: 4000,
  minPairFrequency: 5,
  maxTokenLength: 20,
});
// Generate vocabulary from corpus
// tokenizer.loadCorpusAsArray("data/corpus-13mb.txt");
// tokenizer.generateVocabulary();
tokenizer.loadData({
  pathVocabulary: "vocabulary.json",
  pathMerge: "merge-pairs-js.json",
});
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
const trainingStringArray = fs
  .readFileSync("data/the-verdict.txt", "utf8")
  .split(/(?<= )/);
const trainingTokensArray = [];
for (const word of trainingStringArray) {
  trainingTokensArray.push(tokenizer.encode(word));
}
const numberOfPredictionTaskPerRow = 4;
const stride = 1;
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

// const datasetV3 = new DataSetV3(
//   "data/the-verdict-xsmall.txt",
//   tokenizer,
// );

// const dataLoader = new DataLoaderV2({
//   dataset: datasetV3,
//   batch_size: 2,
//   shuffle: false,
//   numberOfItemsPerRow: numberOfPredictionTaskPerRow,
//   stride: stride,
// });

// const firstBatch = dataLoader.next();
// dataLoader.logLastBatch()
// const secondBatch = dataLoader.next();
// dataLoader.logLastBatch();


const embedding_manager = new EmbeddingManager({
  vocabulary_path: "./vocabulary.json",
  dimensions: 50,
  embeddings_path: "./embeddings.json",
});

// embedding_manager.initializeFromVocabulary("./vocabulary-js.json");
