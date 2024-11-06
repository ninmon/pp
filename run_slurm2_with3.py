#!/home/software/anaconda3/bin/python3
import argparse
import os
import glob
import sys
from pathlib import Path
import subprocess
import re
import io
import math
from concurrent.futures import ProcessPoolExecutor
import time
import tifffile
import pwd 
import grp
import shutil

def is_file_stable(filepath, wait_time=0.8, check_interval=0.4):
    """ Check if a file size remains constant over 'wait_time' seconds."""
    previous_size = -1
    stable_time = 0
    while stable_time < wait_time:
        current_size = filepath.stat().st_size
        if current_size == previous_size:
            stable_time += check_interval
        else:
            stable_time = 0
            previous_size = current_size
        time.sleep(check_interval)
    return True

MATCH_LIST = ["*.tif", "*.tiff"]

def add_args(parser):
    
    # input and output
    parser.add_argument('--input', type=str, default=None, help='Input directory contains .tif / .tiff')
    parser.add_argument('--gain', type=str, default=None, help="Gain reference file in dm4 format")
    parser.add_argument('-o','--output', type=str, default=None, help='Output directory')
    
    # optics parameters
    parser.add_argument("-p", "--pixel_size", type=float, default=1, help="Pixel size in Angstrom")
    parser.add_argument("-v", "--accel_kv", type=float, default=300.0, help="Accelaration voltage in kV")
    parser.add_argument("-c", "--cs_mm", type=float, default=2.70, help="Spherical aberration in mm")
    parser.add_argument("-a", "--amp_contrast", type=float, default=0.07, help="Amplitude contrast")
    
    # motioncor2 parameters
    parser.add_argument("-pat", "--patch", type=int, default=5, help="Patch size")
    parser.add_argument('-b', "--binning", type=int, default=2, help="Binning factor")
    parser.add_argument("-d", "--dose", type=float, default=60.0, help="Dose per frame in e/A^2")

    parser.add_argument("-esamp", "--eer_sampling", type=float, default=2, help="EER sampling mode for MotionCor2")
    parser.add_argument("-efrac", "--eer_fraction", type=float, default=40, help="EER Fractionation for MotionCor2")

    # ctffind5 parameters
    parser.add_argument("-s", "--spectrum_size", type=int, default=512, help="Size of amplitude spectrum to compute")
    parser.add_argument("-rmin", "--min_res", type=float, default=30.0, help="Minimum resolution in Angstrom")
    parser.add_argument("-rmax", "--max_res", type=float, default=5.0, help="Maximum resolution in Angstrom")
    parser.add_argument("-dmin", "--min_defocus", type=float, default=5000.0, help="Minimum defocus in Angstrom")
    parser.add_argument("-dmax", "--max_defocus", type=float, default=50000.0, help="Maximum defocus in Angstrom")
    parser.add_argument("-step", "--defocus_step", type=float, default=100.0, help="Defocus step in Angstrom")

    parser.add_argument("-sc", "--scope_num", type=int, help="BioEM facility microscope number")
    
    return parser

def get_tif_frame_count(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        return len(tif.pages)

def recursive_chown_and_acl(path, uid, gid):
    # 首先更改当前目录的所有权，并为用户添加 ACL 权限
    os.chown(path, uid, gid)
    subprocess.run(["setfacl", "-m", f"user:cryosparc-user:rwx", path], check=True)

    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(path, topdown=True):
        # 更改所有子目录的所有权，并为用户添加 ACL 权限
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.chown(dir_path, uid, gid)
            subprocess.run(["setfacl", "-m", f"user:cryosparc-user:rwx", dir_path], check=True)
        
        # 更改所有文件的所有权，并为用户添加 ACL 权限
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.chown(file_path, uid, gid)
            subprocess.run(["setfacl", "-m", f"user:cryosparc-user:rwx", file_path], check=True)

def submit_to_slurm(job_script):
    cmd = ["sudo -u pp sbatch", job_script]
    ### Submit job to slurm as user "pp" (Running this python script with sudo counts as running it as root, and root cannot submit jobs.)
    subprocess.run(['sudo', '-u', 'pp', 'sbatch', job_script], check=True)


def create_slurm_script(script_path, project_name, tiff_files_chunk, gain_out, args, frame_num, motioncor2_dir, ctffind5_dir, stigma_dir, flag_dir, scope, nums):
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=T{scope}-{nums}-{project_name}\n")
        f.write(f"#SBATCH --gres=gpu:4\n")
        f.write(f"#SBATCH --partition=pp\n")
        f.write(f"#SBATCH --exclusive\n")
        f.write(f"#SBATCH --output=/home/pp/out/1.out\n")
        f.write(f"#SBATCH --error=/home/pp/err/1.err\n")
        f.write(f"sudo /home/pp/conda/pp-1.0/bin/python /home/pp/code/process_tiff_files.py --tiff_files {' '.join(map(str, tiff_files_chunk))} --gain_out {gain_out} "
                f"--binning {args.binning} --patch {args.patch} --dose {args.dose} --pixel_size {args.pixel_size} "
                f"--accel_kv {args.accel_kv} --cs_mm {args.cs_mm} --amp_contrast {args.amp_contrast} --spectrum_size {args.spectrum_size} --eer_fraction {args.eer_fraction} "
                f"--min_res {args.min_res} --max_res {args.max_res} --min_defocus {args.min_defocus} --max_defocus {args.max_defocus} --flag_dir {flag_dir} --eer_sampling {args.eer_sampling} "
                f"--defocus_step {args.defocus_step} --frame_num {frame_num} --motioncor2_dir {motioncor2_dir} --ctffind5_dir {ctffind5_dir} --stigma_dir {stigma_dir} --scope_id {scope}\n")

def main(args):
    ### Get input_dir
    input_dir = Path(args.input) if args.input is not None else Path(os.getcwd())
    print(str(input_dir))


    ### Attempt to acquire scope_num
    if args.input is None and args.scope_num is None:
        if input_dir.parts[2][:5] == "Titan":
            scope = input_dir.parts[2][5]
        else:
            print('Please check current path or specify input path')
            sys.exit(1)
    print(scope)

    scope = int(scope)

    if scope == 3:
        input_dir_data = input_dir / "data"
    else:
        input_dir_data = input_dir
    
    ### Get project name from input_dir path
    project_name = os.path.basename(input_dir)

    ### Get and make output_dir. Get user uid & gid from args.output
    output_dir = Path(args.output) if args.output is not None else input_dir
    if args.output is not None :
        username = output_dir.parts[2]
        try:
            user_info = pwd.getpwnam(username)
            uid = user_info.pw_uid
            gid = user_info.pw_gid
            print(username,uid,gid)
        except Exception as e:
            print(e)
            sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Chown output_dir if exists in args 
    if args.output is not None :
        os.chown(output_dir, uid, gid)   
    
    ### Set gain file fullpath from project name 
    gain_out = str(output_dir / f"{project_name}_gain.mrc")

    ### Convert gain file from dm4 if needed and chown
    if Path(gain_out).exists():
        print(f"converted gain file found")
    elif args.gain is None and scope != 3:
        dm4_files = glob.glob(os.path.join(input_dir, "*.dm4"))
        if dm4_files:
            gain_in = dm4_files[0]
            
            # Copy a backup dm4 file to output_dir
            if args.output is not None:
                dm4_file_copy = output_dir / os.path.basename(gain_in)
                shutil.copy(gain_in, dm4_file_copy)  

            #### Trick: Running this script with sudo will clear the evironment, so we must start from source ~/.bashrc
            subprocess.run('bash -c "source ~/.bashrc && module load pp && dm2mrc {} {}"'.format(gain_in, gain_out), shell=True, check=True)
        else:
            print(f"No gain reference file found, please check agian.")
            sys.exit(1)
    elif args.gain is None and scope == 3:
        gain_files = glob.glob(os.path.join(input_dir, "*.gain"))
        if gain_files:
            gain_in = gain_files[0]
            
            # Copy a backup dm4 file to output_dir
            if args.output is not None:
                gain_file_copy = output_dir / os.path.basename(gain_in)
                shutil.copy(gain_in, gain_file_copy)  
            gain_out_temp = str(output_dir / f"{project_name}_gain_temp.mrc")
            #### Trick: Running this script with sudo will clear the evironment, so we must start from source ~/.bashrc
            subprocess.run('bash -c "source ~/.bashrc && module load pp && tif2mrc {} {}"'.format(gain_in, gain_out_temp), shell=True, check=True)
            subprocess.run('bash -c "source ~/.bashrc && module load pp && source activate eman2 && e2proc2d.py --process math.reciprocal {} {}"'.format(gain_out_temp, gain_out), shell=True, check=True)
        else:
            print(f"No gain reference file found, please check agian.")
            sys.exit(1)
    else:   # If specified gain file
        gain = Path(args.gain)
        gain_in = str(gain)

        # Copy a backup dm4 file to output_dir
        if args.output is not None:
            dm4_file_copy = output_dir / os.path.basename(gain_in)
            shutil.copy(gain_in, dm4_file_copy) 

        if str(gain).endswith('.dm4'):
            subprocess.run('bash -c "source ~/.bashrc && module load pp && dm2mrc {} {}"'.format(gain_in, gain_out), shell=True, check=True)
        elif str(gain).endswith('.gain'):
            gain_out_temp = str(output_dir / f"{project_name}_gain_temp.mrc")
            subprocess.run('bash -c "source ~/.bashrc && module load pp && tif2mrc {} {}"'.format(gain_in, gain_out_temp), shell=True, check=True)
            subprocess.run('bash -c "source ~/.bashrc && module load pp && source activate eman2 && e2proc2d.py --process math.reciprocal {} {}"'.format(gain_out_temp, gain_out), shell=True, check=True)

        # subprocess.run(cmd, check=True)
    if args.output is not None :
        os.chown(gain_out, uid, gid)

    ### Maybe useless
    assert input_dir.exists() and input_dir.is_dir(), f"Input directory {input_dir} does not exist"
    # assert gain.exists() and gain.suffix == ".dm4", f"Gain reference file {gain} does not exist"

    #### Prepare directories and chown them
    motioncor2_dir = output_dir / "motioncor2"
    motioncor2_dir.mkdir(parents=True, exist_ok=True)
    ctffind5_dir = output_dir / "ctffind5"
    ctffind5_dir.mkdir(parents=True, exist_ok=True)
    stigma_dir = input_dir / "stigma"
    stigma_dir.mkdir(parents=True, exist_ok=True)
    script_dir = input_dir / "slurm"
    script_dir.mkdir(parents=True, exist_ok=True)
    flag_dir = input_dir / "flag"
    flag_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None :
        os.chown(flag_dir, uid, gid)
        os.chown(script_dir, uid, gid)
        os.chown(stigma_dir, uid, gid)
        os.chown(ctffind5_dir, uid, gid)
        os.chown(motioncor2_dir, uid, gid)

    #### Number of files per job submission
    chunk_size = 4  
    processed_files = set()
    timeout = 0

    ### Loop for file scanning
    while True:
        ## Scan for tiff files and initialize list by done_flags 
        tiff_files = list(input_dir_data.glob("*.tif")) + list(input_dir_data.glob("*.tiff")) + list(input_dir_data.glob("*.eer"))
        tiff_files = sorted(tiff_files)
        undone_files = []
        for tiff_file in tiff_files:
            done_flag = flag_dir / (tiff_file.name + ".done")
            if not done_flag.exists():
                undone_files.append(tiff_file)

        ## Filter out already processed files
        new_tiff_files = [f for f in undone_files if f not in processed_files]

        if len(new_tiff_files) >= chunk_size:
            tiff_files_chunk = new_tiff_files[:chunk_size]

            # Mark these files as processed
            processed_files.update(tiff_files_chunk)

            # Get movie shape from the first file
            if is_file_stable(tiff_files_chunk[0]) and is_file_stable(tiff_files_chunk[1]) and is_file_stable(tiff_files_chunk[2]) and is_file_stable(tiff_files_chunk[3]):
                print("ready")
            if scope == 3:
                nums = str(tiff_files_chunk[0])[-14:-8] + "," + str(tiff_files_chunk[-3])[-14:-8] + "," + str(tiff_files_chunk[-2])[-14:-8] + "," + str(tiff_files_chunk[-1])[-14:-8]
                print(nums)
            else:
                nums = str(tiff_files_chunk[0])[-8:-4] + "," + str(tiff_files_chunk[-1])[-8:-4]
                print(nums)

            Eer_frac_path = motioncor2_dir / "fraction"
                
            if not Eer_frac_path.exists() and scope == 3:
                
                frame_num = get_tif_frame_count(tiff_files_chunk[0])
                if frame_num == 0:
                    frame_num = get_tif_frame_count(tiff_files_chunk[1])
                if frame_num == 0:            
                    frame_num = get_tif_frame_count(tiff_files_chunk[2])
                if frame_num == 0:                
                    frame_num = get_tif_frame_count(tiff_files_chunk[3])
                if frame_num == 0:    
                    continue
                dose_per_frame = args.dose / frame_num
                print("EER fractionation file do not exists!")
                with open(str(Eer_frac_path), 'w', encoding='utf-8') as file:
                    # 将变量连接起来并用制表符隔开
                    line = f"{frame_num}\t{args.eer_fraction}\t{dose_per_frame}\n"
                    # 将结果写入文件
                    file.write(line)
            else:
                frame_num = get_tif_frame_count(tiff_files_chunk[0])
                if frame_num == 0:
                    frame_num = get_tif_frame_count(tiff_files_chunk[1])
                if frame_num == 0:
                    frame_num = get_tif_frame_count(tiff_files_chunk[2])
                if frame_num == 0:
                    frame_num = get_tif_frame_count(tiff_files_chunk[3])
                if frame_num == 0:
                    continue
                dose_per_frame = args.dose / frame_num

            # Create SLURM script and submit job
            chunk_index = len(processed_files) // chunk_size
            script_path = script_dir / f"slurm_job_{chunk_index}.sh"
            create_slurm_script(script_path, project_name, tiff_files_chunk, gain_out, args, frame_num,str(motioncor2_dir),str(ctffind5_dir),str(stigma_dir),str(flag_dir), scope,nums)
            os.chmod(script_path, 0o755)
            submit_to_slurm(script_path)
            
            if args.output is not None :
                for tiff_chunk_file in tiff_files_chunk:
                # 构造目标文件路径
                    destination = os.path.join(output_dir, os.path.basename(tiff_chunk_file))
                    shutil.copy2(tiff_chunk_file, destination)
                    os.chown(destination, uid, gid)
                        
            timeout = 0        

            # Wait a bit before scanning again
            time.sleep(5)  # Adjust the sleep interval as needed
        
        elif timeout < 8:
            timeout += 1
        
            time.sleep(5)
        elif len(new_tiff_files) > 0:
            tiff_files_chunk = new_tiff_files

            # Mark these files as processed
            processed_files.update(tiff_files_chunk)

            # Get movie shape from the first file
            print(str(tiff_files_chunk[0])[-8:-4] + "-" + str(tiff_files_chunk[-1])[-8:-4])
            frame_num = get_tif_frame_count(tiff_files_chunk[0])
            

            # Create SLURM script and submit job
            chunk_index = len(processed_files) // chunk_size
            script_path = script_dir / f"slurm_job_{chunk_index}.sh"
            create_slurm_script(script_path, project_name, tiff_files_chunk, gain_out, args, frame_num,str(motioncor2_dir),str(ctffind5_dir),str(stigma_dir),str(flag_dir),scope,nums)
            os.chmod(script_path, 0o755)
            submit_to_slurm(script_path)


            if args.output is not None :
                for tiff_chunk_file in tiff_files_chunk:
                # 构造目标文件路径
                    destination = os.path.join(output_dir, os.path.basename(tiff_chunk_file))
                    shutil.copy2(tiff_chunk_file, destination)
                    os.chown(destination, uid, gid)

            timeout = 0        

            # Wait a bit before scanning again
            time.sleep(5)  # Adjust the sleep interval as needed
        else:
            timeout += 1
            if timeout % 12 == 0 :
                print(f"no input, waited {timeout // 12} minute(s)")
            time.sleep(5)
        
        if timeout > 360:
            print(f"No more input, terminating")
            if args.output is not None:
                print("Setting premissions.")
                recursive_chown_and_acl(output_dir, uid, gid)
                print(f"premissions set on directory {output_dir}.")
            sys.exit(1)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
