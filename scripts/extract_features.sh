#!/bin/bash
PROJECT_DIR=..
mkdir -p $PROJECT_DIR/data/embeddings/
python -m processing.extract_features --input_file=$PROJECT_DIR/data/embeddings/data_info.json --state_dict=$PROJECT_DIR/processing/saved/articleBody/1226_160301/checkpoint-epoch30.pth --output_file=$PROJECT_DIR/data/embeddings/embeddings_summary.h5 --config=$PROJECT_DIR/config/article_body.json