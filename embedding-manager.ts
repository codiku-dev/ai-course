import * as tf from "@tensorflow/tfjs";
import fs from "fs";
type Props = {
  vocabulary_path: string;
  dimensions: number;
  embeddings_path: string;
}
export class EmbeddingManager {
  private config: Props;
  private embeddings: tf.Tensor<tf.Rank.R2> | null = null;
  constructor(config: Props) {
    this.config = config;
    this.tryToLoadEmbeddings();
  }

  private tryToLoadEmbeddings() {
    if (!fs.existsSync(this.config.embeddings_path)) {
      console.log("Embeddings file not found, initializing from vocabulary");
      this.initializeFromVocabulary(this.config.vocabulary_path);
    } else {
      console.log("Embeddings file found, loading from file");
      const embeddings = fs.readFileSync(this.config.embeddings_path, 'utf8');
      this.embeddings = JSON.parse(embeddings);
    }
  }
  private initializeFromVocabulary(vocabulary_path: string) {
    const vocabulary: { [token: string]: number } = require(this.config.vocabulary_path);
    const vocabulary_size = Object.keys(vocabulary).length;
    this.embeddings = tf.randomUniform([vocabulary_size, this.config.dimensions]);
    fs.writeFileSync(`embeddings.json`, JSON.stringify(this.embeddings.arraySync(), null, 2));
    return this.embeddings;
  }

  public toString() {
    return this.embeddings?.arraySync().map((row) => row.map((value) => value.toString()).join(" ")).join("\n");
  }

  public getVectorByIndex(index: number): tf.Tensor<tf.Rank.R2> | undefined {
    try {
      return this.embeddings?.gather(index) as tf.Tensor<tf.Rank.R2>;
    } catch (error) {
      console.error("Error getting vector by index: ", error);
      throw error;
    }
  }
}
