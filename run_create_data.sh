MAGRET_DIR="large-corpus"

python create_pretraining_data.py \
  --path=$MAGRET_DIR/data/ \
  --prefix=keras \
  --out_path=$MAGRET_DIR \
  --mode=funcdef \
  --pre=keras_cls_ \
  --nb_snippets=100000 \
  --sparse_adj \
  --regen_vocab
