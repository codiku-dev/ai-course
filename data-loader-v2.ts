import { Rank, Tensor, tensor2d } from "@tensorflow/tfjs";
import { DataSetV2 } from "./dataset-v2";
import { DataSetV3 } from "./dataset-v3";

type Batch = {
  input_samples: number[][];
  target_samples: number[][];
};

type BatchWithTensors = {
  input_samples: Tensor<Rank.R2>;
  target_samples: Tensor<Rank.R2>;
};
type Props = {
  dataset: DataSetV3;
  batch_size: number;
  shuffle?: boolean;
  context_size: number;
  stride: number;
};

export class DataLoader {
  private readonly dataset: DataSetV3;
  private batch_size: number;
  private current_batch_index = 0;

  // Shuffled index list
  private random_start_index_list: number[] = [];
  private max_batch: number = 0
  private stride: number;
  private tensorHeight: number;
  private tensorWidth: number;
  private start_index_input = 0;
  private current_random_start_index = 0;

  private current_batch: Batch = {
    input_samples: [],
    target_samples: []
  };
  constructor({ dataset, batch_size, shuffle = true, context_size, stride }: Props) {
    this.dataset = dataset;
    this.batch_size = batch_size;
    this.tensorWidth = context_size;
    this.stride = stride;
    this.tensorHeight = this.dataset.getLength() / this.tensorWidth;
    this.max_batch = Math.ceil(this.tensorHeight / this.batch_size)

    // Use a shuffled index list to iterate over the dataset
    this.initSampleStartIndexList();
    if (shuffle) {
      this.shuffleStartIndexList();
    }
  }

  next(): BatchWithTensors {
    const input_samples: number[][] = [];
    const target_samples: number[][] = [];

    const encodedTokensArray = this.dataset.getEncodedTokensArray();

    for (let batch_index = 0; batch_index < this.batch_size; batch_index++) { // POur chaque batch

      // 0 + 0x1
      // const start =
      //   this.start_index_input +
      //   batch_index * this.stride;

      // Ou avec un random start index
      const start = this.random_start_index_list[this.current_random_start_index];
      this.current_random_start_index++;

      // 0, 0+4 => (0,4)
      input_samples.push(
        encodedTokensArray.slice(
          start,
          start + this.tensorWidth
        )
      );
      // (1,5)
      target_samples.push(
        encodedTokensArray.slice(
          start + 1,
          start + this.tensorWidth + 1
        )
      );
    }
    this.start_index_input +=
      this.batch_size * this.stride;

    this.current_batch_index++;
    this.current_batch = {
      input_samples: input_samples,
      target_samples: target_samples,
    };
    return {
      input_samples: tensor2d(input_samples, undefined, "int32"),
      target_samples: tensor2d(target_samples, undefined, "int32"),
    };
  }

  // Is there another batch to load ?
  hasNext() {
    return this.current_batch_index < this.max_batch;
  }


  reset() {
    this.current_batch_index = 0;
    this.shuffleStartIndexList();
  }

  // Shuffle Fisher Yates
  private shuffleStartIndexList(): void {
    for (let i = this.random_start_index_list.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.random_start_index_list[i], this.random_start_index_list[j]] = [
        this.random_start_index_list[j],
        this.random_start_index_list[i],
      ];
    }
  }

  private initSampleStartIndexList() {
    // With sliding windows of stride `this.stride`, a start index is valid as
    // long as the input slice [start, start + width] AND the shifted target
    // slice [start + 1, start + width + 1] still fit inside the corpus.
    const datasetLength = this.dataset.getLength();
    const lastValidStart = datasetLength - this.tensorWidth - 1;
    this.random_start_index_list = [];
    for (let start = 0; start <= lastValidStart; start += this.stride) {
      this.random_start_index_list.push(start);
    }
    this.max_batch = Math.ceil(
      this.random_start_index_list.length / this.batch_size
    );
  }

  logLastBatchAsDecoded() {
    const tokenizer = this.dataset.getTokenizer();
    const decodeTokensSeparately = (sample: number[]) =>
      sample.map((tokenId) => tokenizer.decode([tokenId]));

    console.log("--- Batch number: ", this.current_batch_index, " ---");
    console.log("Input samples");
    console.table(this.current_batch.input_samples.map(decodeTokensSeparately));
    console.log("\nTarget samples");
    console.table(this.current_batch.target_samples.map(decodeTokensSeparately));
  }
  logLastBatch() {
    console.log("Input samples");
    console.table(this.current_batch.input_samples);
    console.log("\nTarget samples");
    console.table(this.current_batch.target_samples);
  }
}
