#!/bin/bash
PROJECT_DIR=..
mkdir -p $PROJECT_DIR/data/embeddings/
python -m preprocessing.bert_embeddings --input_file=$PROJECT_DIR/data/hebrew_news_general.csv --output_dir=$PROJECT_DIR/data/embeddings/ --column_name=articleBody --format_file=h5 --save_tokens