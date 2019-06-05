BIFOCALE_DIR="thesis"

python create_pretraining_data.py \
  --path=$BIFOCALE_DIR/data/ \
  --prefix=thesis \
  --out_path=$BIFOCALE_DIR \
  --mode=varname \
  --pre=thesis_varname_ \
  --nb_snippets=100000 \
  --sparse_adj \
  --regen_vocab
