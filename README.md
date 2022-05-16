## MDS-DR

This is the codebase of our ACL22 Findings Paper [Read Top News First: A Document Reordering Approach for Multi-Document News Summarization](https://arxiv.org/abs/2203.10254)

The code is inherited from [PreSumm](https://github.com/nlpyang/PreSumm). 
We add the implenmentation of document reordering. 
Please refer to PreSumm for the implementation of data pre-processing, summarization, and evaluation.




### Step 1. Data Preporation

Set up the dataset name
```shell script
dataset_name=multinews
```

Follow the PreSumm format to prepare for the json data. We provide toy examples under `json_data2/multinews`.


Convert the json format data to torch format, which will be used as the input of BERT. Files are saved to `bert_data2/multinews_doc_cls/`.

```shell script
python preprocess.py -mode format_to_bert_doc -raw_path ../json_data2/${dataset_name} \
        -save_path ../bert_data2/${dataset_name}_doc_cls/ \
        -n_cpus 10 -log_file ../logs/multinews.log -min_src_nsents 1 -doc_separator "unused0"
```

### Step 2. Model Training
Train a documents-level reordering model

```shell script
dataset_name=multinews_doc_cls
python train.py -task ext -mode train_doc -bert_data_path ../bert_data2/${dataset_name} \
-ext_dropout 0.1 -model_path ../models2/${dataset_name}/ \
-lr 2e-3 -visible_gpus 1 -report_every 1000 \
-save_checkpoint_steps 1000 -batch_size 3000 -train_steps 10000 -accum_count 2 -valid_per_steps 1000 \
-log_file ../logs/multinews.log -use_interval true -warmup_steps 2000 -max_pos 512 \
-test_summary_num_sents 11 -block_trigram false -result_path ../results2/${dataset_name}/
```


### Step 3. Model Test
Run the trained model on MDS dataset to evaluate the importance of each document
```shell script
splits=( "train" "valid" "test" )
for split in "${splits[@]}"
do
python train.py  -task ext -mode test_doc -input ${split} -batch_size 1000 -test_batch_size 5 \
-bert_data_path ../bert_data2/${dataset_name}/ -log_file ../logs/multinews.log \
-model_path ../models/${dataset_name} -test_from ../models/${dataset_name}/model_step_10000.pt -sep_optim true \
-use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 \
-result_path ../results2/${dataset_name}/${split}
done
```

