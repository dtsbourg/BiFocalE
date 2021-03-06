BIFOCALE_DIR="large-corpus"
export CUDA_VISIBLE_DEVICES=2

python run_pretraining.py \
  --input_file=$BIFOCALE_DIR/tf_examples_pytorch.tfrecord,$BIFOCALE_DIR/tf_examples_sklearn.tfrecord,$BIFOCALE_DIR/tf_examples_keras.tfrecord \
  --validation_file=$BIFOCALE_DIR/tf_examples_val_pytorch.tfrecord,$BIFOCALE_DIR/tf_examples_val_sklearn.tfrecord,$BIFOCALE_DIR/tf_examples_val_keras.tfrecord \
  --output_dir=$BIFOCALE_DIR/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --save_prediction=True \
  --save_attention=True \
  --bert_config_file=$BIFOCALE_DIR/large_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=200000 \
  --save_checkpoints_steps=10000 \
  --num_warmup_steps=10 \
  --learning_rate=5e-5
