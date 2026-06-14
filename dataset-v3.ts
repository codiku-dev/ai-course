// Data set but this time with tensors

import { Tokenizer } from "./tokenizer";
import fs from "fs";

type Props = {
  filePath: string;
  tokenizer: Tokenizer;
  with_logger?: boolean;
}
export class DataSetV3 {
  private readonly encodedTokensArray: number[] = [];
  private readonly tokenizer: Tokenizer;
  private with_logger: boolean;
  constructor({ filePath, tokenizer, with_logger = false }: Props) {
    this.tokenizer = tokenizer;
    this.with_logger = with_logger;
    this.loadData(filePath);
  }

  private loadData(filePath: string) {
    const text = fs.readFileSync(filePath, "utf8");
    const trainingStringArray = text.split(/(?<= )/);
    if (this.with_logger) {
      console.log("Training string array: ", trainingStringArray);
    }
    for (const word of trainingStringArray) {
      // We flatten the array of tokens to have a single array of tokens
      this.encodedTokensArray.push(...this.tokenizer.encode(word));
    }

    const decodedTokens: string[] = [];
    for (const token of this.encodedTokensArray) {
      decodedTokens.push(this.tokenizer.decode([token]))

    }
    if (this.with_logger) {
      console.log("Encoded tokens: ", this.encodedTokensArray);
      console.log("Decoded tokens: ", decodedTokens);
    }
  }

  public getEncodedTokensArray(): number[] {
    return this.encodedTokensArray;
  }

  public getLength(): number {
    return this.encodedTokensArray.length;
  }

  public getTokenizer(): Tokenizer {
    return this.tokenizer;
  }

}
