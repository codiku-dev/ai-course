import { DataSetV1 } from "./dataset-v1";
import { Tokenizer } from "./tokenizer";

export class DataLoaderV1 {
    private readonly dataset: DataSetV1;
    constructor(filePath: string, tokenizer: Tokenizer, numberOfItemsPerRow: number, stride: number) {
        this.dataset = new DataSetV1(filePath, tokenizer, numberOfItemsPerRow, stride);
    }
}