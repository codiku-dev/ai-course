import { Tokenizer } from "./tokenizer";
import fs from "fs";
export class DataSetV1 {
  private readonly trainingTokensArray: number[] = [];
  private readonly inputIdTensor: number[][] = [];
  private readonly outputIdTensor: number[][] = [];
  private readonly tokenizer: Tokenizer;

  constructor(filePath: string, tokenizer: Tokenizer, numberOfItemsPerRow: number, stride: number) {
    this.tokenizer = tokenizer;

    this.loadData(filePath);

    for (let i = 0; i + numberOfItemsPerRow + 1 <= this.trainingTokensArray.length; i += stride) {
      const inputRow = this.trainingTokensArray.slice(i, i + numberOfItemsPerRow);
      const outputRow = this.trainingTokensArray.slice(i + 1, i + numberOfItemsPerRow + 1);
      this.inputIdTensor.push(inputRow);
      this.outputIdTensor.push(outputRow);
    }
  }

  public getInputRowByIndex(index: number): number[] {
    return this.inputIdTensor[index];
  }

  public getOutputRowByIndex(index: number): number[] {
    return this.outputIdTensor[index];
  }

  public getDecodedInputRowByIndex(index: number): string[] {
    return this.inputIdTensor[index].map(token => this.tokenizer.decode([token]));
  }

  public getDecodedOutputRowByIndex(index: number): string[] {
    return this.outputIdTensor[index].map(token => this.tokenizer.decode([token]));
  }

  private loadData(filePath: string) {
    const text = fs.readFileSync(filePath, "utf8");
    const trainingStringArray = text.split(/(?<= )/)
    for (const word of trainingStringArray) {
      // We flatten the array of tokens to have a single array of tokens
      this.trainingTokensArray.push(...this.tokenizer.encode(word))
    }
  }
  public getLength(): number {
    return this.trainingTokensArray.length;
  }
  public getInputTensor(): number[][] {
    return this.inputIdTensor;
  }
  public toStringInputTensor(): string[][] {
    return this.inputIdTensor.map(row => row.map(token => this.tokenizer.decode([token])));
  }
  public toStringOutputTensor(): string[][] {
    return this.outputIdTensor.map(row => row.map(token => this.tokenizer.decode([token])));
  }
  public getOutputTensor(): number[][] {
    return this.outputIdTensor;
  }
}
