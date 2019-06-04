MAGRET_DIR="sparse"
export CUDA_VISIBLE_DEVICES=0

python run_pretraining.py \
  --input_file=$MAGRET_DIR/tf_examples_mlm.tfrecord \
  --validation_file=$MAGRET_DIR/tf_examples_val_mlm.tfrecord \
  --output_dir=$MAGRET_DIR/pretraining_output-small-300k \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --save_prediction=False \
  --save_attention=True \
  --bert_config_file=$MAGRET_DIR/shallow_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=300000 \
  --save_checkpoints_steps=100000 \
  --num_warmup_steps=10000 \
  --learning_rate=5e-5
