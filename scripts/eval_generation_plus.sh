#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT=$(pwd)

# 模型权重存放目录
CKPT_ROOT="/inspire/hdd/project/exploration-topic/public/ent/TIFO/ckpts"
# 外部数据集存放目录
DATA_ROOT="${PROJ_ROOT}/datasets"
# 结果输出目录
OUTPUT_ROOT="${PROJ_ROOT}/results"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"

# Python 环境设置
python_ext=python3
pip_ext=pip3

test_gen_eval() {
    GEN_EVAL_ROOT="${PROJ_ROOT}/evaluation_gen/gen_eval"

    # run plus inference
    ${python_ext} evaluation_gen/gen_eval/janus_infer4eval_plus.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --ckpt_path ${janus_ckpt_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --n_samples ${n_samples} \
    --rewrite_prompt ${rewrite_prompt} \
    --metadata_file ${GEN_EVAL_ROOT}/prompts/evaluation_metadata.jsonl \
    --janus_model_path ${janus_model_path} \
    --attn_implementation eager \
    --latent_topk ${latent_topk} \
    --latent_attention_layers ${latent_attention_layers} \
    --latent_repeats ${latent_repeats} \
    --latent_cfg_mode ${latent_cfg_mode} \
    ${plus_extra_args}

    # detect objects
    ${python_ext} ${GEN_EVAL_ROOT}/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config ${GEN_EVAL_ROOT}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path ${CKPT_ROOT}

    # accumulate results
    ${python_ext} ${GEN_EVAL_ROOT}/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}


# ================= Inference Arguments =================
pn=1M
model_type=janus_pro_8B
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0

infinity_model_path="${CKPT_ROOT}/infinity_2b_reg.pth"
vae_path="${CKPT_ROOT}/infinity_vae_d32reg.pth"
text_encoder_ckpt="${CKPT_ROOT}/t5xl"
janus_model_path="/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"
janus_ckpt_path="/inspire/hdd/project/exploration-topic/public/ent/NIPS/ckpt/t2i_generation/14208860000.0/ckpt/iter_13999.pth"


# ================= Plus Arguments =================
# 推荐先用较保守配置；如果增强太弱，再提高 latent_topk 或 latent_repeats。
latent_topk=4
latent_attention_layers=4
latent_repeats=1
latent_cfg_mode=shared
plus_extra_args=""
# plus_extra_args="--save_latent_debug"
# plus_extra_args="--keep_filler_tokens"


# ================= Output Arguments =================
out_dir_root="${OUTPUT_ROOT}/eval_results/posttraining-13999"

vae_type=32
cfg=5
tau=1
n_samples=4
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}_plus_topk${latent_topk}_rep${latent_repeats}_${latent_cfg_mode}


# ================= Execution Blocks =================
rewrite_prompt=0
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite
mkdir -p ${out_dir}/results
test_gen_eval
