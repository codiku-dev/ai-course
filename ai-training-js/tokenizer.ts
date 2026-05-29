import fs from 'fs';

type TokenPair = { left: string; right: string };

export class Tokenizer {

    tokens: string[] = [];
    vocabulary: Map<string, number> = new Map();
    reverseVocabulary: Map<number, string> = new Map();
    mergePairs: Array<[string, string]> = [];
    targetVocabSize: number;
    minPairFrequency: number;
    maxTokenLength: number;
    withLogger?: boolean;

    constructor({ withLogger = false, targetVocabSize = 3000, minPairFrequency = 5, maxTokenLength = 20 }: { withLogger: boolean, targetVocabSize: number, minPairFrequency: number, maxTokenLength: number }) {
        this.targetVocabSize = targetVocabSize;
        this.minPairFrequency = minPairFrequency;
        this.maxTokenLength = maxTokenLength;
        this.withLogger = withLogger;
    }

    loadCorpusAsArray(path: string) {
        const corpus = fs.readFileSync(path, 'utf8');
        this.tokens = corpus.replace(/\0/g, '').split('');
        if (this.withLogger) {
            console.log(`Loaded ${this.tokens.length} characters`)
        }
    }

    generateVocabulary() {
        let vocabTokenId = 0
        while (this.vocabulary.size < this.targetVocabSize) {
            if (this.withLogger) {
                console.log(`Generating vocabulary... ${this.vocabulary.size} => Objective is ${this.targetVocabSize}`)
            }
            const pairsWithCounts = this.pairCount()
            const bestPair = this.getBestPair(pairsWithCounts)

            if (bestPair === null) { break };
            if (!bestPair.left || bestPair.left.length + bestPair.right.length > this.maxTokenLength || bestPair.left.length === 0 || bestPair.right.length === 0) break;
            this.mergePairs.push([bestPair.left, bestPair.right])
            this.vocabulary.set(bestPair.left + bestPair.right, vocabTokenId)
            vocabTokenId++
            this.replaceAdjacentPairInTokenSCorpus(bestPair.left, bestPair.right)
        }
        // save vocabulary as json
        const vocabularyAsObject = Object.fromEntries(this.vocabulary);
        const vocabularyMap = new Map<string, number>(Object.entries(vocabularyAsObject) as [string, number][]);
        this.vocabulary = vocabularyMap;
        this.buildReverseVocabulary();
        fs.writeFileSync('vocabulary-js.json', JSON.stringify(vocabularyAsObject, null, 2));
        // save merge pairs as json
        fs.writeFileSync('merge-pairs-js.json', JSON.stringify(this.mergePairs, null, 2));
    }

    private asTokenPair(key: string): TokenPair {
        const separatorIndex = key.indexOf('\0');
        return {
            left: key.slice(0, separatorIndex),
            right: key.slice(separatorIndex + 1),
        };
    }

    private pairCount() {
        const pairsWithCount = new Map<string, number>();
        for (let i = 0; i < this.tokens.length - 1; i++) {
            const left = this.tokens[i];
            const right = this.tokens[i + 1];
            const key = left + '\0' + right;
            const oldPairCount = pairsWithCount.get(key) || 0;
            pairsWithCount.set(key, oldPairCount + 1);
        }
        return pairsWithCount;
    }

    private getBestPair(pairMap: Map<string, number>): TokenPair | null {
        let bestKey = "";
        let bestCount = 0;
        for (const [pair, count] of pairMap.entries()) {
            if (count > bestCount) {
                bestKey = pair
                bestCount = count;
            }
        }
        if (bestCount < this.minPairFrequency) {
            return null
        }
        return this.asTokenPair(bestKey);
    }

    private replaceAdjacentPairInTokenSCorpus(left: string, right: string) {
        const newTokens: string[] = [];
        for (let i = 0; i < this.tokens.length; i++) {
            if (
                i < this.tokens.length - 1 &&
                this.tokens[i] === left &&
                this.tokens[i + 1] === right
            ) {
                newTokens.push(left + right);
                i++
            } else {
                newTokens.push(this.tokens[i]);
            }
        }
        this.tokens = newTokens;
    }

    loadData(config: { pathVocabulary: string, pathMerge: string, pathEmbeddings?: string }) {
        const { pathVocabulary, pathMerge, pathEmbeddings } = config;
        const vocabulary = fs.readFileSync(pathVocabulary, 'utf8');
        const parsedVocabulary = JSON.parse(vocabulary);
        const vocabularyMap = new Map<string, number>(Object.entries(parsedVocabulary) as [string, number][]);
        this.vocabulary = vocabularyMap;
        this.buildReverseVocabulary();
        const mergePairs = fs.readFileSync(pathMerge, 'utf8');
        this.mergePairs = JSON.parse(mergePairs);

        console.log(`Loaded vocabulary containing ${this.vocabulary.size} tokens entries`)
        console.log(`Loaded reverse vocabulary containing ${this.reverseVocabulary.size} tokens entries`)
        console.log(`Loaded merge pairs containing ${this.mergePairs.length} pairs entries`)
        // if (pathEmbeddings) {
        //     const embeddings = fs.readFileSync(pathEmbeddings, 'utf8');
        //     this.embeddings = JSON.parse(embeddings);
        // }
    }



    decode(tokenIdList: number[]): string {
        let decoded = "";
        for (const tokenId of tokenIdList) {
            decoded += this.reverseVocabulary.get(tokenId);
        }
        return decoded;
    }

    private buildReverseVocabulary() {
        for (const [key, value] of this.vocabulary.entries()) {
            this.reverseVocabulary.set(value as number, key);
        }
    }
}
