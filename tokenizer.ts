import fs from 'fs';
import { BASE_VOCABULARY } from './occidental-chars.constant';

type TokenPair = { left: string; right: string };

export const UNKNOWN_TOKEN = '<|unk|>';

export class Tokenizer {

    tokens: string[] = [];
    vocabulary: Map<string, number> = new Map();
    reverse_vocabulary: Map<number, string> = new Map();
    merge_pairs: Map<string, number> = new Map();

    with_logger?: boolean;

    constructor({ with_logger = false }: { with_logger: boolean }) {
        this.with_logger = with_logger;
    }

    loadCorpusAsArray(path: string) {
        const corpus = fs.readFileSync(path, 'utf8');
        this.tokens = corpus.replace(/\0/g, '').split('');
        if (this.with_logger) {
            console.log(`Tokenizer - Loaded ${this.tokens.length} characters`)
        }
    }

    generateVocabulary(config: { target_vocab_size: number, min_pair_frequency: number, max_token_length: number }) {
        const { target_vocab_size, min_pair_frequency, max_token_length } = config;

        let vocab_token_id = 0
        // Reserve the first id for the unknown token
        this.vocabulary.set(UNKNOWN_TOKEN, vocab_token_id)
        vocab_token_id++
        // Seed the vocabulary with the base occidental characters first
        for (const char of BASE_VOCABULARY) {

            if (!this.vocabulary.has(char)) {
                this.vocabulary.set(char, vocab_token_id)
                vocab_token_id++
            }
        }
        while (this.vocabulary.size < target_vocab_size) {
            console.log(`Vocabulary size : ${this.vocabulary.size} => Target is ${target_vocab_size}`)
            const pairs_with_counts = this.pairCount()
            const best_pair = this.getBestPair(pairs_with_counts, max_token_length, min_pair_frequency)

            if (best_pair === null) { break };
            this.merge_pairs.set(best_pair.left + '\0' + best_pair.right, this.merge_pairs.size)
            this.vocabulary.set(best_pair.left + best_pair.right, vocab_token_id)
            vocab_token_id++
            this.replaceAdjacentPairInTokenSCorpus(best_pair.left, best_pair.right)
        }
        // save vocabulary as json
        const vocabulary_as_object = Object.fromEntries(this.vocabulary);
        const vocabulary_map = new Map<string, number>(Object.entries(vocabulary_as_object) as [string, number][]);
        this.vocabulary = vocabulary_map;
        this.buildReverseVocabulary();
        fs.writeFileSync(`vocabulary-js-${new Date().toISOString()}.json`, JSON.stringify(vocabulary_as_object, null, 2));
        // save merge pairs as an associative object { "left\0right": rank }
        const merge_pairs_as_object = Object.fromEntries(this.merge_pairs);
        fs.writeFileSync(`merge-pairs-js-${new Date().toISOString()}.json`, JSON.stringify(merge_pairs_as_object, null, 2));

    }

    private asTokenPair(key: string): TokenPair {
        const separator_index = key.indexOf('\0');
        return {
            left: key.slice(0, separator_index),
            right: key.slice(separator_index + 1),
        };
    }

    private pairCount() {
        const pairs_with_count = new Map<string, number>();
        for (let i = 0; i < this.tokens.length - 1; i++) {
            const left = this.tokens[i];
            const right = this.tokens[i + 1];
            const key = left + '\0' + right;
            const old_pair_count = pairs_with_count.get(key) || 0;
            pairs_with_count.set(key, old_pair_count + 1);
        }
        return pairs_with_count;
    }

    private getBestPair(pair_map: Map<string, number>, max_token_length: number, min_pair_frequency: number): TokenPair | null {
        let best_key = "";
        let best_count = 0;
        for (const [pair, count] of pair_map.entries()) {
            if (count <= best_count) continue;
            const { left, right } = this.asTokenPair(pair);
            // On ignore les paires vides ou qui dépasseraient la longueur max d'un token
            if (!left || !right || left.length + right.length > max_token_length) continue;
            best_key = pair
            best_count = count;
        }
        if (best_count < min_pair_frequency) {
            return null
        }
        return this.asTokenPair(best_key);
    }

    private replaceAdjacentPairInTokenSCorpus(left: string, right: string) {
        const new_tokens: string[] = [];
        for (let i = 0; i < this.tokens.length; i++) {
            if (
                i < this.tokens.length - 1 &&
                this.tokens[i] === left &&
                this.tokens[i + 1] === right
            ) {
                new_tokens.push(left + right);
                i++
            } else {
                new_tokens.push(this.tokens[i]);
            }
        }
        this.tokens = new_tokens;
    }

    loadData(config: { path_vocabulary: string, path_merge: string, path_embeddings?: string }) {
        const { path_vocabulary, path_merge, path_embeddings } = config;
        const vocabulary = fs.readFileSync(path_vocabulary, 'utf8');
        const parsed_vocabulary = JSON.parse(vocabulary);
        const vocabulary_map = new Map<string, number>(Object.entries(parsed_vocabulary) as [string, number][]);
        this.vocabulary = vocabulary_map;
        this.buildReverseVocabulary();
        const merge_pairs = fs.readFileSync(path_merge, 'utf8');
        const parsed_merge_pairs = JSON.parse(merge_pairs);
        this.merge_pairs = new Map<string, number>(Object.entries(parsed_merge_pairs) as [string, number][]);



    }



    decode(token_id_list: number[]): string {
        let decoded = "";
        for (const token_id of token_id_list) {
            decoded += this.reverse_vocabulary.get(token_id) ?? UNKNOWN_TOKEN;
        }

        return decoded;
    }

    encode(text: string): number[] {

        const token_ids: number[] = [];

        // Tokens qui vont évoluer
        const evolutive_pairs_tokens_array = text.split("");

        while (true) {

            let best_pair_index = -1;
            let best_rank = Number.MAX_SAFE_INTEGER;

            // Cherche la meilleure paire actuellement présente

            for (let i = 0; i < evolutive_pairs_tokens_array.length - 1; i++) {

                // Construit la clé de la paire (pour aller la chercher dans le merge_pairs)
                const key =
                    evolutive_pairs_tokens_array[i] +
                    "\0" +
                    evolutive_pairs_tokens_array[i + 1];

                // On récupere son rang
                const rank = this.merge_pairs.get(key);

                // Si rang il y'a et que c'est plus petit que le meilleur rang actuel, on met à jour le meilleur rang et l'index de la paire
                if (
                    rank !== undefined &&
                    rank < best_rank
                ) {
                    best_rank = rank;
                    best_pair_index = i;
                }
            }

            // Arrivé ici, on a parcouru toutes les paires possibles et on a POTENTIELLEMENT
            //  trouvé la meilleure 
            // paire actuellement présente
            // Si on a pas trouvé de meilleure paire, on sort de la boucle
            if (best_pair_index === -1) {
                break;
            }

            // Si on a trouvé une meilleure paire, on la fusionne dans le tableau de tokens
            const merged_token =
                evolutive_pairs_tokens_array[best_pair_index] +
                evolutive_pairs_tokens_array[best_pair_index + 1];

            evolutive_pairs_tokens_array.splice(
                best_pair_index,
                2,
                merged_token
            );
        }

        const unknown_token_id = this.getUnknownTokenId();

        // On parcourt le tableau de tokens avec paires fusionnées intégrées et on créé un tableau de token ids
        for (const token of evolutive_pairs_tokens_array) {
            const token_id =
                this.vocabulary.get(token);

            token_ids.push(
                token_id ?? unknown_token_id
            );
        }




        return token_ids;
    }

    private getUnknownTokenId(): number {
        const unknown_token_id = this.vocabulary.get(UNKNOWN_TOKEN) as number;
        if (unknown_token_id === undefined) {
            throw new Error("Tokenizer - Unknown token (<|unk|>) not found in vocabulary");
        }
        return unknown_token_id;
    }

    private buildReverseVocabulary() {
        for (const [key, value] of this.vocabulary.entries()) {
            this.reverse_vocabulary.set(value as number, key);
        }
    }
}
