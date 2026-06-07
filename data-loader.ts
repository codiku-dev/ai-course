import { Rank, Tensor } from "@tensorflow/tfjs";
import { DataSetV2 } from "./dataset-v2";

type Batch = {
  inputSamples: Tensor<Rank.R2>;
  targetSamples: Tensor<Rank.R2>;
};

type Props = {
  dataset: DataSetV2;
  batch_size: number;
  shuffle?: boolean;
};

export class DataLoader {
  private readonly dataset: DataSetV2;
  private batch_size: number;
  private current_batch_index = 0;
  private current_batch?: Batch;
  private start_index = 0;
  private end_index = 0;
  // Shuffled index list
  private sample_index_list: number[] = [];
  private max_batch: number = 0

  constructor({ dataset, batch_size, shuffle = true }: Props) {
    this.dataset = dataset;
    this.batch_size = batch_size;
    this.max_batch = Math.ceil((dataset.getHeight() / this.batch_size))
    // Use a shuffled index list to iterate over the dataset
    this.initSampleIndexList();
    if (shuffle) {
      this.shuffleIndices();
    }
  }

  next(): Batch {
    this.start_index = this.current_batch_index * this.batch_size;
    // TO make sure we never go out of the data set ( if batch is 10  and dataset is 21 => it's going to crash )

    //  const end_index = start_index + this.batch_size;
    // //imaginons batch size 10 et data set size 22
    // min( 0 + 10 , 21) => 10
    // (10 + 10 , 21) => 20
    // 30 , 21 => 21 => As u can see the end index cannot go further than the length
    this.end_index = Math.min(
      this.start_index + this.batch_size,
      this.dataset.getHeight(),
    );

    // Indices of the samples that belong to this batch (shuffled order)
    const indices = this.sample_index_list.slice(this.start_index, this.end_index);
    // One gather per batch => a single [batch_size, width] tensor for inputs and targets
    this.current_batch = {
      inputSamples: this.dataset.getInputSamplesTensorByIndexes(indices),
      targetSamples: this.dataset.getTargetSamplesTensorByIndexes(indices),
    };
    this.current_batch_index++;
    return this.current_batch;
  }

  // Is there another batch to load ?
  hasNext() {
    return this.current_batch_index < this.max_batch;
  }

  logBatch() {
    if (!this.current_batch) return;
    console.log(`--- BATCH ${this.current_batch_index} ---`);
    console.log("Input samples");
    console.table(this.current_batch.inputSamples.arraySync());
    console.log("\nTarget samples");
    console.table(this.current_batch.targetSamples.arraySync());
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
    for (let i = 0; i < this.dataset.getHeight(); i++) {
      this.sample_index_list[i] = i;
    }
  }
}
