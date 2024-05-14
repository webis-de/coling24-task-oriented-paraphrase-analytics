# Task-Oriented Paraphrase Analytics

Code accompanying the paper *"Task-Oriented Paraphrase Analytics"* at LREC-COLING 2024.

## Organization

* Manual task annotation 
  * Samples and labels can be found here `data/annotations`
* Automatic task classification
  * Serialized models can be found here `data/models`
  * Samples and original corpora with precomputed feature values can be found here `data/corpora`
* Code
  * All used source code can be found in `src`

## Run Instructions

### Install dependencies

```shell
make clean install
```

### Predict tasks with serialized classifier 
```shell
make predict
```

### Retrain and serialize task classifier
```shell
make train
```

## Citation

```bibtex
@InProceedings{gohsen:2024b,
  author =                   {Marcel Gohsen and Matthias Hagen and Martin Potthast and Benno Stein},
  booktitle =                {Joint Conference of the 31st Conference on Computational Linguistics and 16th Conference on Language Resources and Evaluation (LREC-COLING 2024)},
  month =                    may,
  publisher =                {International Committee on Computational Linguistics},
  site =                     {Torino, Italy},
  title =                    {{Task-Oriented Paraphrase Analytics}},
  year =                     2024
}
```