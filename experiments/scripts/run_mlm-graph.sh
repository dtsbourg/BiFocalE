BIFOCALE_DIR="graph_cls"
export CUDA_VISIBLE_DEVICES=2

python run_pretraining.py \
  --input_file=$BIFOCALE_DIR/tf_examples_MSRC_21.tfrecord \
  --validation_file=$BIFOCALE_DIR/tf_examples_val_MSRC_21.tfrecord \
  --output_dir=$BIFOCALE_DIR/pretraining_output-MSRC_21 \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --save_prediction=True \
  --save_attention=True \
  --bert_config_file=$BIFOCALE_DIR/shallow_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=1 \
  --num_train_steps=10000 \
  --save_checkpoints_steps=10000 \
  --num_warmup_steps=10 \
  --learning_rate=5e-5
