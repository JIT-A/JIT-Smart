
# Pretrain and finetune
for prj in `cat project_list.txt` 
do
    prj=$(echo $prj | sed 's/\r//') 
    echo "=================================================================="
    echo "project $prj starting ..."
    echo "Start cross-prj training for project $prj"
    python -m JITFine.concat.run \
    --output_dir=model/jitfine/$prj/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/cross_prj_data/jitfine/$prj/changes_train.pkl data/cross_prj_data/jitfine/$prj/features_train.pkl \
    --eval_data_file data/cross_prj_data/jitfine/$prj/changes_valid.pkl data/cross_prj_data/jitfine/$prj/features_valid.pkl \
    --test_data_file data/cross_prj_data/jitfine/$prj/changes_test.pkl data/cross_prj_data/jitfine/$prj/features_test.pkl \
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
    2>&1| tee model/jitfine/$prj/saved_models_concat/train.log

    echo "=================================================================="
    echo "Start cross-prj inferencing for project $prj"
    python -m JITFine.concat.run --output_dir=model/jitfine/$prj/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/cross_prj_data/jitfine/$prj/changes_train.pkl data/cross_prj_data/jitfine/$prj/features_train.pkl \
    --eval_data_file data/cross_prj_data/jitfine/$prj/changes_valid.pkl data/cross_prj_data/jitfine/$prj/features_valid.pkl \
    --test_data_file data/cross_prj_data/jitfine/$prj/changes_test.pkl data/cross_prj_data/jitfine/$prj/features_test.pkl \
    --epoch 50 --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 256 \
    --eval_batch_size 24 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds \
    --buggy_line_filepath=data/cross_prj_data/jitfine/$prj/changes_complete_buggy_line_level.pkl \
    --seed 42 \
    2>&1 | tee model/jitfine/$prj/saved_models_concat/test.log
    
    
    echo "project $prj starting ending..."
done
echo "Finish all"
echo "=================================================================="

