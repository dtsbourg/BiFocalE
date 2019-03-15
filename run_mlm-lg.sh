MAGRET_DIR="large-corpus"
export CUDA_VISIBLE_DEVICES=1

PREFIX="_mlm_split_magret"

PREFIX_KERAS="keras${PREFIX}"
PREFIX_SKLEARN="sklearn${PREFIX}"
PREFIX_PYTORCH="pytorch${PREFIX}"
PREFIX_ANSIBLE="ansible${PREFIX}"
PREFIX_REQUESTS="requests${PREFIX}"
PREFIX_DJANGO="django${PREFIX}"
PREFIX_HTTPIE="httpie${PREFIX}"
PREFIX_YT="youtube-dl${PREFIX}"
PREFIX_FLASK="flask${PREFIX}"
PREFIX_BERT="bert${PREFIX}"

IN_KERAS="$MAGRET_DIR/tf_examples${PREFIX_KERAS}.tfrecord"
IN_SKLEARN="$MAGRET_DIR/tf_examples${PREFIX_SKLEARN}.tfrecord"
IN_PYTORCH="$MAGRET_DIR/tf_examples${PREFIX_PYTORCH}.tfrecord"
IN_ANSIBLE="$MAGRET_DIR/tf_examples${PREFIX_ANSIBLE}.tfrecord"
IN_REQUESTS="$MAGRET_DIR/tf_examples${PREFIX_REQUESTS}.tfrecord"
IN_DJANGO="$MAGRET_DIR/tf_examples${PREFIX_DJANGO}.tfrecord"
IN_HTTPIE="$MAGRET_DIR/tf_examples${PREFIX_HTTPIE}.tfrecord"
IN_YT="$MAGRET_DIR/tf_examples${PREFIX_YT}.tfrecord"
IN_FLASK="$MAGRET_DIR/tf_examples${PREFIX_FLASK}.tfrecord"
IN_BERT="$MAGRET_DIR/tf_examples${PREFIX_BERT}.tfrecord"


IN_VAL_KERAS="$MAGRET_DIR/tf_examples_val${PREFIX_KERAS}.tfrecord"
IN_VAL_SKLEARN="$MAGRET_DIR/tf_examples_val${PREFIX_SKLEARN}.tfrecord"
IN_VAL_PYTORCH="$MAGRET_DIR/tf_examples_val${PREFIX_PYTORCH}.tfrecord"
IN_VAL_ANSIBLE="$MAGRET_DIR/tf_examples_val${PREFIX_ANSIBLE}.tfrecord"
IN_VAL_REQUESTS="$MAGRET_DIR/tf_examples_val${PREFIX_REQUESTS}.tfrecord"
IN_VAL_DJANGO="$MAGRET_DIR/tf_examples_val${PREFIX_DJANGO}.tfrecord"
IN_VAL_HTTPIE="$MAGRET_DIR/tf_examples_val${PREFIX_HTTPIE}.tfrecord"
IN_VAL_YT="$MAGRET_DIR/tf_examples_val${PREFIX_YT}.tfrecord"
IN_VAL_FLASK="$MAGRET_DIR/tf_examples_val${PREFIX_FLASK}.tfrecord"
IN_VAL_BERT="$MAGRET_DIR/tf_examples_val${PREFIX_BERT}.tfrecord"

python run_pretraining.py \
  --input_file=$IN_KERAS,$IN_SKLEARN,$IN_PYTORCH,$IN_ANSIBLE,$IN_REQUESTS,$IN_DJANGO,$IN_HTTPIE,$IN_YT,$IN_FLASK,$IN_BERT \
  --validation_file=$IN_VAL_KERAS,$IN_VAL_SKLEARN,$IN_VAL_PYTORCH,$IN_VAL_ANSIBLE,$IN_VAL_REQUESTS,$IN_VAL_DJANGO,$IN_VAL_HTTPIE,$IN_VAL_YT,$IN_VAL_FLASK,$IN_VAL_BERT \
  --output_dir=$MAGRET_DIR/pretraining_output-100 \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --save_prediction=False \
  --save_attention=True \
  --bert_config_file=$MAGRET_DIR/shallow_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=100 \
  --save_checkpoints_steps=100000 \
  --num_warmup_steps=10000 \
  --learning_rate=5e-5
