MAGRET_DIR="mid-corpus"
export CUDA_VISIBLE_DEVICES=0

PREFIX="_mlm_split_magret"

PREFIX_KERAS="keras${PREFIX}"
PREFIX_SKLEARN="sklearn${PREFIX}"
PREFIX_PYTORCH="pytorch${PREFIX}"

IN_KERAS="$MAGRET_DIR/tf_examples${PREFIX_KERAS}.tfrecord"
IN_SKLEARN="$MAGRET_DIR/tf_examples${PREFIX_SKLEARN}.tfrecord"
IN_PYTORCH="$MAGRET_DIR/tf_examples${PREFIX_PYTORCH}.tfrecord"

IN_VAL_KERAS="$MAGRET_DIR/tf_examples_val${PREFIX_KERAS}.tfrecord"
IN_VAL_SKLEARN="$MAGRET_DIR/tf_examples_val${PREFIX_SKLEARN}.tfrecord"
IN_VAL_PYTORCH="$MAGRET_DIR/tf_examples_val${PREFIX_PYTORCH}.tfrecord"

python run_pretraining.py \
  --input_file=$IN_KERAS,$IN_SKLEARN,$IN_PYTORCH \
  --validation_file=$IN_VAL_KERAS,$IN_VAL_SKLEARN \
  --output_dir=$MAGRET_DIR/pretraining_output-1 \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --save_prediction=False \
  --save_attention=True \
  --bert_config_file=$MAGRET_DIR/shallow_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=1000 \
  --save_checkpoints_steps=100000 \
  --num_warmup_steps=100 \
  --learning_rate=5e-5
