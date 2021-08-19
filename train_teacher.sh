id="teacher"
export LC_ALL="en_US.UTF-8"

python3 trainstudent.py --id $id \
    --fraction 0.01\
    --obj_dropout 0 \
    --caption_model aoa \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.1 \
    --input_json data/newcocotalk.json \
    --input_label_h5 data/newcocotalk_label.h5 \
    --input_fc_dir ../cocobu_fc_labelled\
    --input_att_dir  ../cocobu_att_labelled  \
    --seq_per_img 5 \
    --batch_size 16 \
    --beam_size 2 \
    --remove_bad_endings 1\
    --block_trigrams 1\
    --decoding_constraint 1\
    --learning_rate 2e-4 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/log_$id  \
    $start_from \
    --save_checkpoint_every 2000 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 100 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3 \
    --n_augs 1 \
    --topk 10 \
    --grad_clip 2 \
> $id".txt"
