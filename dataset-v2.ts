// Data set but this time with tensors

import { Tokenizer } from "./tokenizer";
import { Tensor, Rank, tensor2d } from "@tensorflow/tfjs";
import fs from "fs";

export class DataSetV2 {
    private readonly trainingTokensArray: number[] = [];
    private readonly inputIdTensor: Tensor<Rank.R2>;
    private readonly outputIdTensor: Tensor<Rank.R2>;
    private readonly tokenizer: Tokenizer;

    constructor(filePath: string, tokenizer: Tokenizer, numberOfItemsPerRow: number, stride: number) {
        this.tokenizer = tokenizer;

        this.loadData(filePath);

        const inputMatrix: number[][] = [];
        const outputMatrix: number[][] = [];
        for (let i = 0; i + numberOfItemsPerRow + 1 <= this.trainingTokensArray.length; i += stride) {
            inputMatrix.push(this.trainingTokensArray.slice(i, i + numberOfItemsPerRow));
            outputMatrix.push(this.trainingTokensArray.slice(i + 1, i + numberOfItemsPerRow + 1));
        }
        // One 2D tensor for all inputs and one for all targets (the course's x and y)
        this.inputIdTensor = tensor2d(inputMatrix);
        this.outputIdTensor = tensor2d(outputMatrix);
    }

    private loadData(filePath: string) {
        const text = fs.readFileSync(filePath, "utf8");
        const trainingStringArray = text.split(/(?<= )/)
        for (const word of trainingStringArray) {
            // We flatten the array of tokens to have a single array of tokens
            this.trainingTokensArray.push(...this.tokenizer.encode(word))
        }
    }

    public getInputTensor(): Tensor<Rank.R2> {
        return this.inputIdTensor;
    }

    public getOutputTensor(): Tensor<Rank.R2> {
        return this.outputIdTensor;
    }

    public getInputRowByIndex(index: number): number[] {
        return this.inputIdTensor.arraySync()[index];
    }

    public getOutputRowByIndex(index: number): number[] {
        return this.outputIdTensor.arraySync()[index];
    }

    public getDecodedInputRowByIndex(index: number): string[] {
        return this.getInputRowByIndex(index).map(token => this.tokenizer.decode([token]));
    }

    public getDecodedOutputRowByIndex(index: number): string[] {
        return this.getOutputRowByIndex(index).map(token => this.tokenizer.decode([token]));
    }

    public getShape(): [number, number] {
        return this.inputIdTensor.shape;
    }

    public getLength(): number {
        return this.trainingTokensArray.length;
    }
}
