
# Pretrain and finetune
for prj in `cat project_list.txt` 
do
    prj=$(echo $prj | sed 's/\r//')
    echo "=================================================================="
    echo "project $prj starting ..."
    echo "Start cross-prj training for project $prj"
    python -m JITSmart.concat.run \
    --output_dir=model/jitsmart/$prj/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/cross_prj_data/jitsmart/$prj/changes_train.pkl data/cross_prj_data/jitsmart/$prj/features_train.pkl \
    --eval_data_file data/cross_prj_data/jitsmart/$prj/changes_valid.pkl data/cross_prj_data/jitsmart/$prj/features_valid.pkl \
    --test_data_file data/cross_prj_data/jitsmart/$prj/changes_test.pkl data/cross_prj_data/jitsmart/$prj/features_test.pkl \
    --epoch 50 \
    --max_seq_length 512  \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-5  \
    --max_grad_norm 1.0  \
    --evaluate_during_training \
    --feature_size 14 \
    --seed 42 \
    --patience 10 \
    --max_codeline_length 256 \
    --max_codeline_token_length 64 \
    --buggy_lines_file data/cross_prj_data/jitsmart/$prj/train_buggy_commit_lines_df.pkl data/cross_prj_data/jitsmart/$prj/valid_buggy_commit_lines_df.pkl data/cross_prj_data/jitsmart/$prj/test_buggy_commit_lines_df.pkl \
    2>&1| tee model/jitsmart/$prj/saved_models_concat/train.log

    echo "=================================================================="
    echo "Start cross-prj inferencing for project $prj"
    python -m JITSmart.concat.run --output_dir=model/jitsmart/$prj/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/cross_prj_data/jitsmart/$prj/changes_train.pkl data/cross_prj_data/jitsmart/$prj/features_train.pkl \
    --eval_data_file data/cross_prj_data/jitsmart/$prj/changes_valid.pkl data/cross_prj_data/jitsmart/$prj/features_valid.pkl \
    --test_data_file data/cross_prj_data/jitsmart/$prj/changes_test.pkl data/cross_prj_data/jitsmart/$prj/features_test.pkl \
    --epoch 50 --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 256 \
    --eval_batch_size 24 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds \
    --buggy_line_filepath=data/cross_prj_data/jitsmart/$prj/test_buggy_commit_lines_df.pkl \
    --seed 42 \
    --max_codeline_length 256 \
    --max_codeline_token_length 64 \
    --buggy_lines_file data/cross_prj_data/jitsmart/$prj/train_buggy_commit_lines_df.pkl data/cross_prj_data/jitsmart/$prj/valid_buggy_commit_lines_df.pkl data/cross_prj_data/jitsmart/$prj/test_buggy_commit_lines_df.pkl \
    2>&1 | tee model/jitsmart/$prj/saved_models_concat/test.log
    
    
    echo "project $prj starting ending..."
done
echo "Finish all"
echo "=================================================================="

