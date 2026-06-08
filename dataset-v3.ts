// Data set but this time with tensors

import { Tokenizer } from "./tokenizer";
import fs from "fs";

export class DataSetV3 {
  private readonly encodedTokensArray: number[] = [];
  private readonly tokenizer: Tokenizer;

  constructor(
    filePath: string,
    tokenizer: Tokenizer,
  ) {
    this.tokenizer = tokenizer;
    this.loadData(filePath);
  }

  private loadData(filePath: string) {
    const text = fs.readFileSync(filePath, "utf8");
    const trainingStringArray = text.split(/(?<= )/);
    console.log("Training string array: ", trainingStringArray);
    for (const word of trainingStringArray) {
      // We flatten the array of tokens to have a single array of tokens
      this.encodedTokensArray.push(...this.tokenizer.encode(word));
    }

    const decodedTokens: string[] = [];
    for (const token of this.encodedTokensArray) {
      decodedTokens.push(this.tokenizer.decode([token]))

    }
    console.log("Encoded tokens: ", this.encodedTokensArray);
    console.log("Decoded tokens: ", decodedTokens);
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
