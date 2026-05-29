import { Tokenizer } from "./tokenizer";

const tokenizer = new Tokenizer({ withLogger: true, targetVocabSize: 1000, minPairFrequency: 5, maxTokenLength: 20 });
// Generate vocabulary from corpus
// tokenizer.loadCorpusAsArray("../corpus-17mb.txt");
// tokenizer.generateVocabulary();
tokenizer.loadData({ pathVocabulary: "vocabulary-js.json", pathMerge: "merge-pairs-js.json" });

console.log(tokenizer.decode([1556, 8, 762, 8, 2966, 8, 441]));