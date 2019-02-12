MAGRET_DIR="sparse"

python create_pretraining_data.py \
  --path=$MAGRET_DIR/data/ \
  --prefix=keras \
  --out_path=$MAGRET_DIR \
  --mode=funcdef \
  --pre=sparse_fname2_ \
  --nb_snippets=100000 \
  --sparse_adj \
  #--regen_vocab
