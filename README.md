# BiFocalE

> Learning Representations of Code from Structure and Context

## Introduction

BiFocalE is a deep encoder architecture which embeds snippets of source code. Specifically, it takes into account the graph structure of the piece of code at hand through its AST. 

## Preparing data

At a high level, the model expects for each snippet of code:

* a textual representation
* a structured representation

In this work, we derive both from the Abstract Syntax Tree (AST). Some pre-generated data is available in datasets like [py-150k](https://eth-sri.github.io/py150) or [js-150k](https://eth-sri.github.io/js150). One can also generate their own ASTs with tools like [semantic](https://github.com/github/semantic) (examples coming soon).

Concretely, the model expects the following input format:

#### For text

One file per snippet, for example `xxx_java_tk.txt`:

```
[CLS] CompilationUnit PackageDeclaration ClassDeclaration ReferenceType Annotation ElementArrayValue MethodDeclaration savedInstanceState ReferenceType StatementExpression SuperMethodInvocation StatementExpression setContentView StatementExpression setupList MethodDeclaration v ReferenceType StatementExpression findViewById setAlpha MethodDeclaration v ReferenceType StatementExpression findViewById setAlpha MethodDeclaration listId
```

> By convention, validation files are denominated with `_val`

#### For structure

One file per AST graph. While both sparse and dense representations are supported, we encourage the use of the former for efficiency purposes. Each graph is stored as a `.mtx` file, with the same index as the corresponding token file. See [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html) for more information about this file format.

## Pre-training models

### Generating pre-training data

The model is usually pre-trained in a semi-supervised fashion. For this purpose, we take the input training snippets and process them to generate a large amount of training data for this task. Specifically, a fraction of the input tokens is masked so the model can attempt to reconstruct them.

To run this, a `run_prepare_xxx.sh` script is run. An example is provided below:

```
BIFOCALE_DIR="xxx"
PREFIX="abc"
VOCAB="vocab.txt"

python prepare_pretraining_data.py \
  --input_file=$BIFOCALE_DIR/${PREFIX}_tk.txt \
  --output_file=$BIFOCALE_DIR/tf_examples${SUFFIX}.tfrecord \
  --vocab_file=$BIFOCALE_DIR/$VOCAB \
  --adj_file=$BIFOCALE_DIR/adj/ \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --masked_lm_prob=0.15 \
  --random_seed=1009 \
  --dupe_factor=50 \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --is_training=True
```

### Pre-training

To run the pre-training procedure, a `run_xxx.sh` script is made available, for example:

```
BIFOCALE_DIR="xxx"
export CUDA_VISIBLE_DEVICES=3

python run_pretraining.py \
  --input_file=$BIFOCALE_DIR/tf_examples.tfrecord \
  --validation_file=$BIFOCALE_DIR/tf_examples_val.tfrecord \
  --output_dir=$BIFOCALE_DIR/pretraining_output-xxx \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --save_prediction=False \
  --save_attention=False \
  --bert_config_file=$BIFOCALE_DIR/config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=100000 \
  --save_checkpoints_steps=50000 \
  --num_warmup_steps=10000 \
  --learning_rate=5e-5
```

## Fine-tuning

The pre-trained model can then be specialized to run downstream supervised tasks. Here again, a set of training and testing labels must be made available. The labels should be placed in a file `xxx-labels.txt` where the line index corresponds to that of the input data.

```
BIFOCALE_DIR="xxx-method"
PREFIX="xxx"
PRETRAIN_DIR="xxx"
PREDIR="xxx"

export CUDA_VISIBLE_DEVICES=4

python classifier.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=True \
  --max_nb_preds=1000 \
  --task_name=methodname \
  --label_vocab=$BIFOCALE_DIR/xxx-vocab-labels-thresh.txt \
  --vocab_file=$BIFOCALE_DIR/$PREDIR/java-vocab.txt \
  --train_file=$BIFOCALE_DIR/$PREDIR/${PREFIX}_tk.txt \
  --train_labels=$BIFOCALE_DIR/$PREDIR/xxx-train-labels.txt \
  --train_adj=$BIFOCALE_DIR/$PREDIR \
  --eval_file=$BIFOCALE_DIR/$PREDIR/${PREFIX}_tk_val.txt \
  --eval_labels=$BIFOCALE_DIR/$PREDIR/xxx-val-labels.txt \
  --eval_adj=$BIFOCALE_DIR/$PREDIR \
  --data_dir=$BIFOCALE_DIR \
  --output_dir=$BIFOCALE_DIR/cls_output \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=1000 \
  --save_checkpoints_steps=10000 \
  --bert_config_file=$BIFOCALE_DIR/config.json \
  --sparse_adj=True \
  --adj_prefix=${PREFIX} \
  --clean_data=True \
  --init_checkpoint=$PRETRAIN_DIR/pretraining_output/model.ckpt
```

## Released models

> TODO


## Tips and caveats

* The model hyper-parameters can be updated in `model-config.json` files.
* A vocabulary file must be generated and provided to the model.

In case of issues please use the "Issues" tab to contact the authors.

## Credit

The base of this implementation was built around the work of [Devlin et al.](https://arxiv.org/abs/1810.04805), and specifically their [implementation of BERT in Tensforflow](https://github.com/google-research/bert). Recognizing the parallels between our architecture and that of BERT, along with the quality of their training infrastructure and methodology, we bootstrapped upon their own implementation. Credit to the project is left where credit is due.
