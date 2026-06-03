import fs from 'fs';
import { BASE_VOCABULARY } from './occidental-chars.constant';

type TokenPair = { left: string; right: string };

export const UNKNOWN_TOKEN = '<|unk|>';

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
            console.log(`Tokenizer - Loaded ${this.tokens.length} characters`)
        }
    }

    generateVocabulary() {
        if (this.withLogger) {
            console.log("Tokenizer - Generating vocabulary...");
        }
        let vocabTokenId = 0
        // Reserve the first id for the unknown token
        this.vocabulary.set(UNKNOWN_TOKEN, vocabTokenId)
        vocabTokenId++
        // Seed the vocabulary with the base occidental characters first
        for (const char of BASE_VOCABULARY) {

            if (!this.vocabulary.has(char)) {
                this.vocabulary.set(char, vocabTokenId)
                vocabTokenId++
            }
        }
        while (this.vocabulary.size < this.targetVocabSize) {
            console.log(`Vocabulary size : ${this.vocabulary.size} => Target is ${this.targetVocabSize}`)
            const pairsWithCounts = this.pairCount()
            const bestPair = this.getBestPair(pairsWithCounts)

            if (bestPair === null) { break };
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
        fs.writeFileSync(`vocabulary-js-${new Date().toISOString()}.json`, JSON.stringify(vocabularyAsObject, null, 2));
        // save merge pairs as an associative object { "left\0right": rank }
        const mergePairsAsObject = Object.fromEntries(this.mergePairs);
        fs.writeFileSync(`merge-pairs-js-${new Date().toISOString()}.json`, JSON.stringify(mergePairsAsObject, null, 2));
        if (this.withLogger) {
            console.log(`Tokenizer - Generated vocabulary containing ${this.vocabulary.size} tokens entries`)
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
            if (count <= bestCount) continue;
            const { left, right } = this.asTokenPair(pair);
            // On ignore les paires vides ou qui dépasseraient la longueur max d'un token
            if (!left || !right || left.length + right.length > this.maxTokenLength) continue;
            bestKey = pair
            bestCount = count;
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
                console.log(`Tokenizer - Loaded vocabulary containing ${this.vocabulary.size} tokens entries`)
            }
        }

    }



    decode(tokenIdList: number[]): string {
        let decoded = "";
        for (const tokenId of tokenIdList) {
            decoded += this.reverseVocabulary.get(tokenId) ?? UNKNOWN_TOKEN;
        }
        if (this.withLogger) {
            console.log("Tokenizer - decode() result :", decoded);
            console.log("Detail of each token id :");
            for (const tokenId of tokenIdList) {
                console.log(`- ${tokenId} : "${this.reverseVocabulary.get(tokenId)}"`);
            }

        }
        return decoded;
    }

    encode(text: string): number[] {
        console.time("encode()");

        const tokenIds: number[] = [];

        // Tokens qui vont évoluer
        const evolutivePairsTokensArray = text.split("");

        while (true) {

            let bestPairIndex = -1;
            let bestRank = Number.MAX_SAFE_INTEGER;

            // Cherche la meilleure paire actuellement présente

            for (let i = 0; i < evolutivePairsTokensArray.length - 1; i++) {

                // Construit la clé de la paire (pour aller la chercher dans le mergePairs)
                const key =
                    evolutivePairsTokensArray[i] +
                    "\0" +
                    evolutivePairsTokensArray[i + 1];

                // On récupere son rang
                const rank = this.mergePairs.get(key);

                // Si rang il y'a et que c'est plus petit que le meilleur rang actuel, on met à jour le meilleur rang et l'index de la paire
                if (
                    rank !== undefined &&
                    rank < bestRank
                ) {
                    bestRank = rank;
                    bestPairIndex = i;
                }
            }

            // Arrivé ici, on a parcouru toutes les paires possibles et on a POTENTIELLEMENT
            //  trouvé la meilleure 
            // paire actuellement présente
            // Si on a pas trouvé de meilleure paire, on sort de la boucle
            if (bestPairIndex === -1) {
                break;
            }

            // Si on a trouvé une meilleure paire, on la fusionne dans le tableau de tokens
            const mergedToken =
                evolutivePairsTokensArray[bestPairIndex] +
                evolutivePairsTokensArray[bestPairIndex + 1];

            evolutivePairsTokensArray.splice(
                bestPairIndex,
                2,
                mergedToken
            );
        }

        const unknownTokenId = this.getUnknownTokenId();

        // On parcourt le tableau de tokens avec paires fusionnées intégrées et on créé un tableau de token ids
        for (const token of evolutivePairsTokensArray) {
            const tokenId =
                this.vocabulary.get(token);

            tokenIds.push(
                tokenId ?? unknownTokenId
            );
        }

        if (this.withLogger) {
            console.log(`Tokenizer - encode("${text}") : ${tokenIds}`);
            console.log("Detail of each token id :");
            for (const tokenId of tokenIds) {
                console.log(`- ${tokenId} : "${this.reverseVocabulary.get(tokenId)}"`);
            }
        }

        console.timeEnd("encode()");

        return tokenIds;
    }

    private getUnknownTokenId(): number {
        const unknownTokenId = this.vocabulary.get(UNKNOWN_TOKEN) as number;
        if (unknownTokenId === undefined) {
            throw new Error("Tokenizer - Unknown token (<|unk|>) not found in vocabulary");
        }
        return unknownTokenId;
    }

    private buildReverseVocabulary() {
        for (const [key, value] of this.vocabulary.entries()) {
            this.reverseVocabulary.set(value as number, key);
        }
    }
}
