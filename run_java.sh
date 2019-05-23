MAGRET_DIR="java-64"
export CUDA_VISIBLE_DEVICES=3

python run_pretraining.py \
  --input_file=$MAGRET_DIR/tf_examples.tfrecord \
  --validation_file=$MAGRET_DIR/tf_examples_val.tfrecord \
  --output_dir=$MAGRET_DIR/pretraining_output-java \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --save_prediction=False \
  --save_attention=False \
  --bert_config_file=$MAGRET_DIR/shallow_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=100000 \
  --save_checkpoints_steps=50000 \
  --num_warmup_steps=10000 \
  --learning_rate=5e-5
