import { DataSetV3 } from "./dataset-v3";
import { Tensor, tensor2d } from "@tensorflow/tfjs"
import { Rank } from "@tensorflow/tfjs";
type Batch = {
  input_samples: Tensor<Rank.R2>;
  target_samples: Tensor<Rank.R2>;
};

type Props = {
  dataset: DataSetV3;
  batch_size: number;
  shuffle?: boolean;
  number_items_per_row: number;
  stride: number;
  drop_last: boolean
};

export class DataLoader {
  private readonly dataset: DataSetV3;
  private batch_size: number;
  private current_batch_index = 0;
  private drop_last: boolean = true
  // Shuffled index list
  private random_start_index_list: number[] = [];
  private max_batch: number = 0
  private stride: number;
  private tensor_height: number;
  private tensor_width: number;
  private start_index_input = 0;
  private current_random_start_index = 0;

  private current_batch: Batch = {
    input_samples: tensor2d([[], []]),
    target_samples: tensor2d([[], []]),
  };
  constructor({ dataset, batch_size, shuffle = true, number_items_per_row, stride, drop_last }: Props) {
    this.dataset = dataset;
    this.batch_size = batch_size;
    this.tensor_width = number_items_per_row;
    this.stride = stride;
    this.tensor_height = this.dataset.getLength() / this.tensor_width;
    // we ceil because 
    this.max_batch = drop_last ? Math.ceil(this.tensor_height / this.batch_size) : Math.floor(this.tensor_height / this.batch_size)
    this.drop_last = drop_last

    // Use a shuffled index list to iterate over the dataset
    this.initSampleStartIndexList();
    if (shuffle) {
      this.shuffleStartIndexList();
    }
  }

  next(): Batch {
    const input_samples: number[][] = [];
    const target_samples: number[][] = [];

    const encoded_tokens_array = this.dataset.getEncodedTokensArray();

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
        encoded_tokens_array.slice(
          start,
          start + this.tensor_width
        )
      );
      // (1,5)
      target_samples.push(
        encoded_tokens_array.slice(
          start + 1,
          start + this.tensor_width + 1
        )
      );
    }
    this.start_index_input +=
      this.batch_size * this.stride;

    this.current_batch_index++;
    this.current_batch = {
      input_samples: tensor2d(input_samples),
      target_samples: tensor2d(target_samples),
    };
    return this.current_batch
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
    const lastValidStart = datasetLength - this.tensor_width - 1;
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
    console.table(this.current_batch.input_samples.arraySync().map(decodeTokensSeparately));
    console.log("\nTarget samples");
    console.table(this.current_batch.target_samples.arraySync().map(decodeTokensSeparately));
  }

}
