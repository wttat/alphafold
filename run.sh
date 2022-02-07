#!/bin/bash
# Description: AlphaFold non-docker version
# Author: Sanjay Kumar Srikakulam
# by WTTAT 

usage() {
        echo ""
        echo "Please make sure all required parameters are given"
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-f <fasta_paths>       Path to a FASTA file containing sequence. If a FASTA file contains multiple sequences, then it will be folded as a multimer"
        echo "-t <max_template_date> Maximum template release date to consider (ISO-8601 format - i.e. YYYY-MM-DD). Important if folding historical test sets"
        echo "Optional Parameters:"
        echo "-m <model_preset>     Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model (default: 'monomer')"
        echo "-c <db_preset>        Choose preset MSA database configuration - smaller genetic database config (reduced_dbs) or full genetic database config (full_dbs) (default: 'full_dbs')"
        echo "-l <is_prokaryote_list>    Optional for multimer system, not used by the single chain system. A boolean specifying true where the target complex is from a prokaryote, and false where it is not, or where the origin is unknown. This value determine the pairing method for the MSA (default: 'None')"
        echo "-p <use_precomputed_msas> Whether to read MSAs that have been written to disk. WARNING: This will not check if the sequence, database or configuration have changed (default: 'false')"
        echo "-r <run_relax> Whether to run the final relaxation step on the predicted models. Turning relax off might result in predictions with distracting stereochemical violations but might help in case you are having issues with the relaxation stage."
        echo "-b <benchmark>        Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many proteins (default: 'false')"
        echo ""
        exit 1
}

while getopts ":f:t:m:c:l:p:r:b" i; do
        case "${i}" in
        f)
                fasta_paths=$OPTARG
        ;;
        t)
                max_template_date=$OPTARG
        ;;
        m)
                model_preset=$OPTARG
        ;;
        c)
                db_preset=$OPTARG
        ;;
        l)
                is_prokaryote_list=$OPTARG
        ;;
        p)
                use_precomputed_msas=$OPTARG
        ;;
        r)
                run_relax=$OPTARG
        ;;
        b)
                benchmark='false'
        ;;
        esac
done

echo "BATCH_BUCKET : $BATCH_BUCKET"
echo "REGION : $REGION"
echo "fasta_paths : $fasta_paths"
echo "max_template_date : $max_template_date"
echo "model_preset : $model_preset"
echo "db_preset : $db_preset"
echo "is_prokaryote_list : $is_prokaryote_list"

pwd

# Parse input and set defaults
if [[ "$fasta_paths" == "" || "$max_template_date" == "" ]] ; then
    usage
fi

if [[ "$model_preset" == "" ]] ; then
    model_preset="monomer"
fi

if [[ "$model_preset" != "monomer" && "$model_preset" != "monomer_casp14" && "$model_preset" != "monomer_ptm" && "$model_preset" != "multimer" ]] ; then
    echo "Unknown model preset! Using default ('monomer')"
    model_preset="monomer"
fi

if [[ "$db_preset" == "" ]] ; then
    db_preset="full_dbs"
fi

if [[ "$db_preset" != "full_dbs" && "$db_preset" != "reduced_dbs" ]] ; then
    echo "Unknown database preset! Using default ('full_dbs')"
    db_preset="full_dbs"
fi

if [[ "$use_precomputed_msas" == "" ]] ; then
    use_precomputed_msas="false"
fi

if [[ "$is_prokaryote_list" == "" ]] ; then
    is_prokaryote_list="false"
fi

if [[ "$is_prokaryote_list" != "true" && "$is_prokaryote_list" != "false" ]] ; then
    echo "Unknown is_prokaryote_list preset! Using default ('false')"
    is_prokaryote_list="false"
fi

echo "model_preset reset: $model_preset"
echo "db_preset reset: $db_preset"
echo "is_prokaryote_list reset: $is_prokaryote_list"

# This bash script looks for the run_alphafold.py script in its current working directory, if it does not exist then exits
current_working_dir=$(pwd)
alphafold_script="$current_working_dir/run_alphafold.py"

# if [ ! -f "$alphafold_script" ]; then
#     echo "Alphafold python script $alphafold_script does not exist."
#     exit 1
# fi

# Export ENVIRONMENT variables and set CUDA devices for use
# CUDA GPU control
# export CUDA_VISIBLE_DEVICES=-1
# if [[ "$use_gpu" == true ]] ; then
#     export CUDA_VISIBLE_DEVICES=0

#     if [[ "$gpu_devices" ]] ; then
#         export CUDA_VISIBLE_DEVICES=$gpu_devices
#     fi
# fi

# OpenMM threads control
# if [[ "$openmm_threads" ]] ; then
#     export OPENMM_CPU_THREADS=$openmm_threads
# fi

# This part set in batch env

# # TensorFlow control
# export TF_FORCE_UNIFIED_MEMORY='1'

# # JAX control
# export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'

# dataset in Fsx for lustre
data_dir="/fsx/dataset"

# Path and user config (change me if required)
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"
uniprot_database_path="$data_dir/uniprot/uniprot.fasta"
mgnify_database_path="$data_dir/mgnify/mgy_clusters_2018_12.fa"
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
small_bfd_database_path="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
uniclust30_database_path="$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
pdb70_database_path="$data_dir/pdb70/pdb70"
pdb_seqres_database_path="$data_dir/pdb_seqres/pdb_seqres.txt"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"

# Binary path (change me if required)
hhblits_binary_path=$(which hhblits)
hhsearch_binary_path=$(which hhsearch)
jackhmmer_binary_path=$(which jackhmmer)
kalign_binary_path=$(which kalign)

# download fasta file from S3
echo "start downloading"
aws s3 cp s3://$BATCH_BUCKET/$INPUT_PREFIX/$fasta_paths ./ --region $REGION

output_dir="/app/output/"

# get vCPU
vcpu=$[$(curl -s $ECS_CONTAINER_METADATA_URI | jq '.Limits.CPU')/1024]
echo "get vCPU : $vcpu"

use_gpu_relax='true'

command_args="--run_relax=$run_relax --use_gpu_relax=$use_gpu_relax --vcpu=$vcpu --fasta_paths=$fasta_paths --output_dir=$output_dir --benchmark=$benchmark --max_template_date=$max_template_date --db_preset=$db_preset --model_preset=$model_preset --use_precomputed_msas=$use_precomputed_msas --logtostderr"

database_paths="--uniref90_database_path=$uniref90_database_path --mgnify_database_path=$mgnify_database_path --data_dir=$data_dir --template_mmcif_dir=$template_mmcif_dir --obsolete_pdbs_path=$obsolete_pdbs_path"

binary_paths="--hhblits_binary_path=$hhblits_binary_path --hhsearch_binary_path=$hhsearch_binary_path --jackhmmer_binary_path=$jackhmmer_binary_path --kalign_binary_path=$kalign_binary_path"

if [[ $model_preset == "multimer" ]]; then
	database_paths="$database_paths --uniprot_database_path=$uniprot_database_path --pdb_seqres_database_path=$pdb_seqres_database_path"
else
	database_paths="$database_paths --pdb70_database_path=$pdb70_database_path"
fi

if [[ "$db_preset" == "reduced_dbs" ]]; then
	database_paths="$database_paths --small_bfd_database_path=$small_bfd_database_path"
else
	database_paths="$database_paths --uniclust30_database_path=$uniclust30_database_path --bfd_database_path=$bfd_database_path"
fi

if [[ $is_prokaryote_list ]]; then
	command_args="$command_args --is_prokaryote_list=$is_prokaryote_list"
fi

echo "command_args: $command_args"
echo "database_paths: $database_paths"
echo "binary_paths: $binary_paths"

# Run AlphaFold with required parameters
echo "start running af2"
# $(python $alphafold_script $binary_paths $database_paths $command_args)
$(python $alphafold_script $binary_paths $database_paths $command_args)

echo "start ziping"
fasta_name=${fasta_paths%.*}

cd $output_dir
tar -zcvf $fasta_name.tar.gz $fasta_name/

echo "start uploading"
aws s3 sync $output_dir/$fasta_name s3://$BATCH_BUCKET/$OUTPUT_PREFIX/$fasta_name  --region $REGION

# add metadata
aws s3 cp $output_dir/$fasta_name.tar.gz s3://$BATCH_BUCKET/$OUTPUT_PREFIX/  --metadata {'"id"':'"'$id'"'} --region $REGION

echo "all done"