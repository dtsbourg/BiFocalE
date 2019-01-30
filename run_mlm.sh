MAGRET_DIR="cls_magret"
export CUDA_VISIBLE_DEVICES=0

python run_pretraining.py \
  --input_file=$MAGRET_DIR/tf_examples.tfrecord \
  --validation_file=$MAGRET_DIR/tf_examples_single.tfrecord \
  --output_dir=$MAGRET_DIR/pretraining_output \
  --do_train=False \
  --do_eval=True \
  --do_predict=True \
  --save_prediction=True \
  --save_attention=True \
  --bert_config_file=$MAGRET_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=100000 \
  --save_checkpoints_steps=10000 \
  --num_warmup_steps=10 \
  --learning_rate=1e-5
