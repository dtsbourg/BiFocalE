BERT_BASE_DIR="uncased_L-12_H-768_A-12"
CUSTOM_BERT_DIR="split_bert_with_multival"

python run_pretraining.py \
  --input_file=$CUSTOM_BERT_DIR/tf_examples.tfrecord \
  --validation_file=$CUSTOM_BERT_DIR/tf_examples_val.tfrecord \
  --output_dir=$CUSTOM_BERT_DIR/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --save_attention=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=1 \
  --num_train_steps=100000 \
  --num_warmup_steps=10 \
  --learning_rate=1e-5
