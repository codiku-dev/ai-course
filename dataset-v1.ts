import { Tokenizer } from "./tokenizer";
import fs from "fs";
export class DataSet {
  private readonly trainingTokensArray: number[][] = [];
  private readonly inputIdMatrix: number[][][] = [];
  private readonly outputIdMatrix: number[][][] = [];
  private readonly tokenizer: Tokenizer;

  constructor(filePath: string, tokenizer: Tokenizer, numberOfItemsPerRow: number, shiftBy: number) {
    this.tokenizer = tokenizer;

    this.loadData(filePath);

    for (let i = 0; i < this.trainingTokensArray.length; i += numberOfItemsPerRow) {
      const inputRow = this.trainingTokensArray.slice(i, i + numberOfItemsPerRow);
      const outputRow = this.trainingTokensArray.slice(i + shiftBy, i + shiftBy + numberOfItemsPerRow);
      this.inputIdMatrix.push(inputRow);
      this.outputIdMatrix.push(outputRow);
    }
  }

  public getInputRowByIndex(index: number): number[][] {
    return this.inputIdMatrix[index];
  }

  public getOutputRowByIndex(index: number): number[][] {
    return this.outputIdMatrix[index];
  }

  public getDecodedInputRowByIndex(index: number): string[] {
    return this.inputIdMatrix[index].map(row => this.tokenizer.decode(row));
  }

  public getDecodedOutputRowByIndex(index: number): string[] {
    return this.outputIdMatrix[index].map(row => this.tokenizer.decode(row));
  }

  private loadData(filePath: string) {
    const text = fs.readFileSync(filePath, "utf8");
    const trainingStringArray = text.split(/(?<= )/)
    for (const word of trainingStringArray) {
      this.trainingTokensArray.push(this.tokenizer.encode(word))
    }
  }
}
