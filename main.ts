import { Tokenizer } from "./tokenizer";

const tokenizer = new Tokenizer({ withLogger: true, targetVocabSize: 4000, minPairFrequency: 5, maxTokenLength: 20 });
// Generate vocabulary from corpus
tokenizer.loadCorpusAsArray("data/corpus-13mb.txt");
tokenizer.generateVocabulary();
// tokenizer.loadData({ pathVocabulary: "vocabulary-js.json", pathMerge: "merge-pairs-js.json" });
// tokenizer.decode([1635, 15])
// tokenizer.encode("hello")