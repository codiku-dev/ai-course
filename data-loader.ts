import { DataSetV1 } from "./dataset-v1";
import { DataSetV2 } from "./dataset-v2";

type Batch = {
  inputSamples: number[][];
  targetSamples: number[][];
};
export class DataLoader {
  private readonly dataset: DataSetV2;
  private batch_size: number;
  private current_batch_index = 0;
  private current_batch: Batch = { inputSamples: [], targetSamples: [] };
  private start_index = 0;
  private end_index = 0;
  // Shuffled index list
  private sample_index_list: number[] = [];

  constructor(dataset: DataSetV2, batch_size: number, shuffle = true) {
    this.dataset = dataset;
    this.batch_size = batch_size;
    this.initSampleIndexList();
    if (shuffle) {
      this.shuffleIndices();
    }
  }

  next(): { inputSamples: number[][]; targetSamples: number[][] } {
    const batch: { inputSamples: number[][]; targetSamples: number[][] } = {
      inputSamples: [],
      targetSamples: [],
    };
    this.start_index = this.current_batch_index * this.batch_size;
    // TO make sure we never go out of the data set ( if batch is 10  and dataset is 21 => it's going to crash )

    //  const end_index = start_index + this.batch_size;
    this.end_index = Math.min(
      this.start_index + this.batch_size,
      this.dataset.getLength(),
    );

    for (let i = this.start_index; i < this.end_index; i++) {
      batch.inputSamples.push(
        this.dataset.getInputSampleByIndex(this.sample_index_list[i]),
      );
      batch.targetSamples.push(
        this.dataset.getTargetSampleByIndex(this.sample_index_list[i]),
      );
    }

    this.current_batch = batch;
    this.current_batch_index++;
    return this.current_batch;
  }

  hasNext() {
    return (
      (this.dataset.getLength() / (this.current_batch_index + 1)) *
        this.batch_size >
      0
    );
  }

  log() {
    console.log(`--- BATCH ${this.current_batch_index} ---`);
    console.log("Input samples");
    console.table(this.current_batch.inputSamples);
    console.log("\nTarget samples");
    console.table(this.current_batch.targetSamples);
  }

  reset() {
    this.current_batch_index = 0;
    this.shuffleIndices();
  }

  // Shuffle Fisher Yates
  private shuffleIndices(): void {
    for (let i = this.sample_index_list.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.sample_index_list[i], this.sample_index_list[j]] = [
        this.sample_index_list[j],
        this.sample_index_list[i],
      ];
    }
  }

  private initSampleIndexList() {
    for (let i = 0; i < this.dataset.getLength(); i++) {
      this.sample_index_list[i] = i;
    }
  }
}
