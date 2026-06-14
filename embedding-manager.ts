import { Tensor, Rank, randomUniform, tensor2d } from "@tensorflow/tfjs";
import fs from "fs";
type Props = {
  dimensions: number;
}
export class EmbeddingManager {
  private embeddings: Tensor<Rank.R2> | null = null;

  private dimensions: number;

  constructor({ dimensions }: Props) {
    this.dimensions = dimensions;
  }
  loadExistingEmbeddings(embeddings_file_path: string) {
    const embeddings = fs.readFileSync(embeddings_file_path, 'utf8');
    this.embeddings = tensor2d(JSON.parse(embeddings));
    if (this.embeddings.shape[1] !== this.dimensions) {
      throw new Error("Embeddings dimensions do not match");
    }
    return this.embeddings;
  }

  initializeEmbeddings({ vocabulary_file_path, output_file_path }: { vocabulary_file_path: string, output_file_path: string, dimensions: number }) {
    const vocabulary: { [token: string]: number } = require(vocabulary_file_path);
    const vocabulary_size = Object.keys(vocabulary).length;
    this.embeddings = randomUniform([vocabulary_size, this.dimensions]);
    fs.writeFileSync(output_file_path, JSON.stringify(this.embeddings.arraySync(), null, 2));
    return this.embeddings;
  }

  public getVectorByIndex(index: number): Tensor<Rank.R2> | undefined {
    try {
      return this.embeddings?.gather(index) as Tensor<Rank.R2>;
    } catch (error) {
      console.error("Error getting vector by index: ", error);
      throw error;
    }
  }
  getEmbeddings(): Tensor<Rank.R2> {
    if (!this.embeddings) {
      throw new Error("Embeddings not initialized");
    }
    return this.embeddings;
  }
}
