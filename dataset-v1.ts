import { Tokenizer } from "./tokenizer";
import fs from "fs";
export class DataSetV1 {
  private readonly trainingTokensArray: number[] = [];
  private readonly inputSamples: number[][] = [];
  private readonly targetSamples: number[][] = [];
  private readonly tokenizer: Tokenizer;

  constructor(
    filePath: string,
    tokenizer: Tokenizer,
    numberOfItemsPerRow: number,
    stride: number,
  ) {
    this.tokenizer = tokenizer;

    this.loadData(filePath);

    for (
      let i = 0;
      i + numberOfItemsPerRow + 1 <= this.trainingTokensArray.length;
      i += stride
    ) {
      const inputRow = this.trainingTokensArray.slice(
        i,
        i + numberOfItemsPerRow,
      );
      const outputRow = this.trainingTokensArray.slice(
        i + 1,
        i + numberOfItemsPerRow + 1,
      );
      this.inputSamples.push(inputRow);
      this.targetSamples.push(outputRow);
    }
  }

  public getInputSampleByIndex(index: number): number[] {
    return this.inputSamples[index];
  }

  public getTargetSampleByIndex(index: number): number[] {
    return this.targetSamples[index];
  }

  public getDecodedInputSampleByIndex(index: number): string[] {
    return this.inputSamples[index].map((token) =>
      this.tokenizer.decode([token]),
    );
  }

  public getDecodedTargetSampleByIndex(index: number): string[] {
    return this.targetSamples[index].map((token) =>
      this.tokenizer.decode([token]),
    );
  }

  private loadData(filePath: string) {
    const text = fs.readFileSync(filePath, "utf8");
    const trainingStringArray = text.split(/(?<= )/);
    for (const word of trainingStringArray) {
      // We flatten the array of tokens to have a single array of tokens
      this.trainingTokensArray.push(...this.tokenizer.encode(word));
    }
  }
  public getLength(): number {
    return this.trainingTokensArray.length;
  }
  public getInputSamples(): number[][] {
    return this.inputSamples;
  }
  public toStringInputSamples(): string[][] {
    return this.inputSamples.map((row) =>
      row.map((token) => this.tokenizer.decode([token])),
    );
  }
  public toStringTargetSamples(): string[][] {
    return this.targetSamples.map((row) =>
      row.map((token) => this.tokenizer.decode([token])),
    );
  }
  public getTargetSamples(): number[][] {
    return this.targetSamples;
  }
}
