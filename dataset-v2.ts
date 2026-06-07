// Data set but this time with tensors

import { Tokenizer } from "./tokenizer";
import { Tensor, Rank, tensor2d } from "@tensorflow/tfjs";
import fs from "fs";

export class DataSetV2 {
  private readonly trainingTokensArray: number[] = [];
  private readonly inputSamples: Tensor<Rank.R2>;
  private readonly targetSamples: Tensor<Rank.R2>;
  private readonly tokenizer: Tokenizer;

  constructor(
    filePath: string,
    tokenizer: Tokenizer,
    numberOfItemsPerRow: number,
    stride: number,
  ) {
    this.tokenizer = tokenizer;

    this.loadData(filePath);

    const inputSample: number[][] = [];
    const targetSample: number[][] = [];
    for (
      let i = 0;
      i + numberOfItemsPerRow + 1 <= this.trainingTokensArray.length;
      i += stride
    ) {
      inputSample.push(
        this.trainingTokensArray.slice(i, i + numberOfItemsPerRow),
      );
      targetSample.push(
        this.trainingTokensArray.slice(i + 1, i + numberOfItemsPerRow + 1),
      );
    }
    // One 2D tensor for all inputs and one for all targets (the course's x and y)
    this.inputSamples = tensor2d(inputSample);
    this.targetSamples = tensor2d(targetSample);
  }

  private loadData(filePath: string) {
    const text = fs.readFileSync(filePath, "utf8");
    const trainingStringArray = text.split(/(?<= )/);
    for (const word of trainingStringArray) {
      // We flatten the array of tokens to have a single array of tokens
      this.trainingTokensArray.push(...this.tokenizer.encode(word));
    }
  }

  public getInputSamples(): Tensor<Rank.R2> {
    return this.inputSamples;
  }
  public getInputSamplesTensorByIndexes(indexes: number[]): Tensor<Rank.R2> {
    return this.inputSamples.gather(indexes);
  }
  public getTargetSamplesTensorByIndexes(indexes: number[]): Tensor<Rank.R2> {
    return this.targetSamples.gather(indexes);
  }
  public getTargetSamples(): Tensor<Rank.R2> {
    return this.targetSamples;
  }

  public getDecodedInputSamples(): string[][] {
    return this.getInputValues().map((row) =>
      row.map((token) => this.tokenizer.decode([token])),
    );
  }

  public getDecodedTargetSamples(): string[][] {
    return this.getTargetValues().map((row) =>
      row.map((token) => this.tokenizer.decode([token])),
    );
  }

  public getInputSampleByIndex(index: number): number[] {
    return this.inputSamples.arraySync()[index];
  }

  public getTargetSampleByIndex(index: number): number[] {
    return this.targetSamples.arraySync()[index];
  }

  public getDecodedInputSampleByIndex(index: number): string[] {
    return this.getInputSampleByIndex(index).map((token) =>
      this.tokenizer.decode([token]),
    );
  }

  public getDecodedTargetSampleByIndex(index: number): string[] {
    return this.getTargetSampleByIndex(index).map((token) =>
      this.tokenizer.decode([token]),
    );
  }

  public getInputValues(): number[][] {
    return this.inputSamples.arraySync();
  }
  public getTargetValues(): number[][] {
    return this.targetSamples.arraySync();
  }

  public getHeight(): number {
    return this.inputSamples.shape[0];
  }

  public getWidth(): number {
    return this.inputSamples.shape[1];
  }

  public getTokensCount(): number {
    return this.trainingTokensArray.length;
  }

  public getValues(): { inputs: number[][]; targets: number[][] } {
    return {
      inputs: this.getInputValues(),
      targets: this.getTargetValues(),
    };
  }

  public getDecodedValues(): { inputs: string[][]; targets: string[][] } {
    return {
      inputs: this.getDecodedInputSamples(),
      targets: this.getDecodedTargetSamples(),
    };
  }

  public log() {
    const values = this.getValues();
    console.log("Inputs");
    console.table(values.inputs);
    console.log("\nTargets");
    console.table(values.targets);
  }

  public logDecoded() {
    const values = this.getDecodedValues();
    console.log("Inputs");
    console.table(values.inputs);
    console.log("\nTargets");
    console.table(values.targets);
  }
}
