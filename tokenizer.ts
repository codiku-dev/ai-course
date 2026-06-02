import fs from 'fs';

type TokenPair = { left: string; right: string };

export class Tokenizer {

    tokens: string[] = [];
    vocabulary: Map<string, number> = new Map();
    reverseVocabulary: Map<number, string> = new Map();
    mergePairs: Map<string, number> = new Map();
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
            console.log(`Tokenizer: Loaded ${this.tokens.length} characters`)
        }
    }

    generateVocabulary() {
        if (this.withLogger) {
            console.log("Tokenizer: Generating vocabulary...");
        }
        let vocabTokenId = 0
        while (this.vocabulary.size < this.targetVocabSize) {
            const pairsWithCounts = this.pairCount()
            const bestPair = this.getBestPair(pairsWithCounts)

            if (bestPair === null) { break };
            if (!bestPair.left || bestPair.left.length + bestPair.right.length > this.maxTokenLength || bestPair.left.length === 0 || bestPair.right.length === 0) break;
            this.mergePairs.set(bestPair.left + '\0' + bestPair.right, this.mergePairs.size)
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
        // save merge pairs as an associative object { "left\0right": rank }
        const mergePairsAsObject = Object.fromEntries(this.mergePairs);
        fs.writeFileSync('merge-pairs-js.json', JSON.stringify(mergePairsAsObject, null, 2));
        if (this.withLogger) {
            console.log(`Tokenizer: Generated vocabulary containing ${this.vocabulary.size} tokens entries`)
        }
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
        const parsedMergePairs = JSON.parse(mergePairs);
        this.mergePairs = new Map<string, number>(Object.entries(parsedMergePairs) as [string, number][]);

        if (this.withLogger) {
            if (this.vocabulary.size > 0) {
                console.log(`Tokenizer: Loaded vocabulary containing ${this.vocabulary.size} tokens entries`)
            }
        }

    }



    decode(tokenIdList: number[]): string {
        console.time("decode()");
        let decoded = "";
        for (const tokenId of tokenIdList) {
            decoded += this.reverseVocabulary.get(tokenId);
        }
        if (this.withLogger) {
            console.log("Tokenizer: decode() result :", decoded);
            console.timeEnd("decode()");

        }
        return decoded;
    }

    encode(text: string): number[] {
        console.time("encode()");
        const tokenIds: number[] = [];
        //The tokens that will evolve
        const newTokens = text.split('');
        // On parcours les paires de merge (la Map garde l'ordre d'insertion = ordre des index)
        // Looping trough each pair
        for (const key of this.mergePairs.keys()) {
            // On récupere la partie gauche et droite de la paire de merge
            const { left, right } = this.asTokenPair(key);

            // On parcours l'input de texte
            for (let j = 0; j < newTokens.length; j++) {
                // On regarde si on trouve cote a cote la paire de merge
                if (newTokens[j] === left && newTokens[j + 1] === right) {
                    // On remplace la paire de merge par le nouveau token
                    newTokens.splice(j, 2, left + right);
                }
            }
        }
        // On parcours les tokens et on
        for (const token of newTokens) {
            tokenIds.push(this.vocabulary.get(token) || 0);
        }
        if (this.withLogger) {
            console.log("Tokenizer: encode() - ", tokenIds);
            console.timeEnd("encode()");
        }
        return tokenIds;
    }

    private buildReverseVocabulary() {
        for (const [key, value] of this.vocabulary.entries()) {
            this.reverseVocabulary.set(value as number, key);
        }
    }
}
