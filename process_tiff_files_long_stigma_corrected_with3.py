
import os
import subprocess
import argparse
from pathlib import Path
import multiprocessing
import pandas as pd
import io
import math
os.environ['PATH'] += '/home/software/cuda-12.6.2/bin'

ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ld_library_path + ':/home/software/cuda-12.6.2/lib64'

os.environ['PATH'] += ':/usr/local/bin:/home/software/MotionCor2_1.6.4'
os.environ['PATH'] += ':/usr/local/bin:/home/software/ctffind-5.0.2'

TEMPLATE_BASH_SCRIPT = Path("/home/pp/code/template.sh")

CTFFIND5_COLUMNS = ['Micrograph Number', 'Defocus 1 [Angstroms]', 'Defocus 2 [Angstroms]', 'Azimuth of Astigmatism',
    'Additional Phase Shift [Radians]', 'Cross Correlation', 'CTF Rings Fit Spacing [Angstroms]',
    'Estimated Tilt Axis Angle', 'Estimated Tilt Angle', 'Estimated Sample Thickness [Angstroms]']

def read_ctffind5_txt(txt_file):
    data_lines = []
    with open(txt_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('#') and line:  # 跳过以#开头的注释行和空行
                data_lines.append(line)
    data = data_lines[-1]
    df = pd.read_csv(io.StringIO(data), delim_whitespace=True, header=None, names=CTFFIND5_COLUMNS)
    return df 

def calculate_stigma(defocus_u, defocus_v, stigma_angle, scope, obj_stigma_x=0, obj_stigma_y=0):
    # Constants from the original Perl script
    if scope == 1:    
        x_TEMstigma_step = 38.01
        y_TEMstigma_step = 36.81
        y_TEMstigma_angle = -23.61
        x_TEMstigma_angle = -68.67
    elif scope == 2:
        x_TEMstigma_step = 37.01
        y_TEMstigma_step = 35.58
        x_TEMstigma_angle = 63.98
        y_TEMstigma_angle = 19.06
    elif scope == 3:
        x_TEMstigma_step = 37.01
        y_TEMstigma_step = 35.58
        x_TEMstigma_angle = -19.06
        y_TEMstigma_angle = -63.98

      
    # Calculations for Lx
    x_angle_diff = x_TEMstigma_angle - stigma_angle
    kx1 = math.sin(math.radians(x_angle_diff)) / math.cos(math.radians(x_angle_diff))
    x_angle_diff += 90
    kx2 = math.sin(math.radians(x_angle_diff)) / math.cos(math.radians(x_angle_diff))

    Lx_sub1 = math.sqrt((kx1 * kx1 + 1) / (defocus_v * defocus_v + defocus_u * defocus_u * kx1 * kx1))
    Lx_sub2 = math.sqrt((kx2 * kx2 + 1) / (defocus_v * defocus_v + defocus_u * defocus_u * kx2 * kx2))
    Lx = 0.0001 * defocus_u * defocus_v * (Lx_sub1 - Lx_sub2) / x_TEMstigma_step

    # Calculations for Ly
    y_angle_diff = y_TEMstigma_angle - stigma_angle
    ky1 = math.sin(math.radians(y_angle_diff)) / math.cos(math.radians(y_angle_diff))
    y_angle_diff += 90
    ky2 = math.sin(math.radians(y_angle_diff)) / math.cos(math.radians(y_angle_diff))

    Ly_sub1 = math.sqrt((ky1 * ky1 + 1) / (defocus_v * defocus_v + defocus_u * defocus_u * ky1 * ky1))
    Ly_sub2 = math.sqrt((ky2 * ky2 + 1) / (defocus_v * defocus_v + defocus_u * defocus_u * ky2 * ky2))
    Ly = 0.0001 * defocus_u * defocus_v * (Ly_sub1 - Ly_sub2) / y_TEMstigma_step

    # Calculate new stigma values
    new_stigma_x = obj_stigma_x + Lx
    new_stigma_x = -1 * new_stigma_x
    new_stigma_y = obj_stigma_y + Ly
    new_stigma_y = -1 * new_stigma_y

    return "{:.5f}".format(new_stigma_x), "{:.5f}".format(new_stigma_y)

def process_tiff_file(tiff_file, gain_out, motioncor2_dir, ctffind5_dir, stigma_dir, args, frame_num, gpu_id, scope, flag_dir):
    filename_without_extension = os.path.splitext(os.path.basename(tiff_file))[0]
    inputfile = os.path.basename(tiff_file)
    mrc_file = motioncor2_dir / (filename_without_extension + ".mrc")
    print(os.environ['PATH'])
    # Run MotionCor2
    if scope == 1 or scope == 2:    
        cmd = [
        "/home/software/MotionCor2_1.6.4/MotionCor2",
        "-InTiff", str(tiff_file), "-Gain", str(gain_out), "-OutMrc", str(mrc_file),
        "-FtBin", str(args.binning), "-Patch", f"{args.patch} {args.patch}",
        "-FmDose", str(args.dose / frame_num), "-PixSize", str(args.pixel_size),
        "-kV", str(args.accel_kv), "-Gpu", str(gpu_id)
    ]
    elif scope == 3:
        Eer_frac_path = motioncor2_dir / "fraction"
        cmd = [
        "/home/software/MotionCor2_1.6.4/MotionCor2",
        "-InEer", str(tiff_file), "-Gain", str(gain_out), "-OutMrc", str(mrc_file),
        "-FtBin", str(args.binning), "-EerSampling", str(args.eer_sampling), "-FmIntFile", str(Eer_frac_path), "-Patch", f"{args.patch} {args.patch}",
        "-PixSize", str(args.pixel_size),
        "-kV", str(args.accel_kv), "-Gpu", str(gpu_id)
    ]
    print(f"Run command: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Run bash script for ctffind5
    freq_mrc_file = ctffind5_dir / (filename_without_extension + ".mrc")
    pixel_size_ctf = args.pixel_size * args.binning
    cmd = [
        "bash", str(TEMPLATE_BASH_SCRIPT),
        "-i", str(mrc_file), "-o", str(freq_mrc_file),
        "-p", str(pixel_size_ctf), "-v", str(args.accel_kv),
        "-c", str(args.cs_mm), "-a", str(args.amp_contrast),
        "-s", str(args.spectrum_size), "-rmin", str(args.min_res),
        "-rmax", str(args.max_res), "-dmin", str(args.min_defocus),
        "-dmax", str(args.max_defocus), "-step", str(args.defocus_step)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Read ctffind5 output and calculate stigma
    txt_file = ctffind5_dir / (filename_without_extension + ".txt")
    ctf_params = read_ctffind5_txt(txt_file)

    
    defocus_u = ctf_params['Defocus 1 [Angstroms]'].values[0]
    defocus_v = ctf_params['Defocus 2 [Angstroms]'].values[0]
    defocus_u_s = "{:.1f}".format(defocus_u)
    defocus_v_s = "{:.1f}".format(defocus_v)
    delta_def = defocus_u - defocus_v
    delta_def_s = "{:.1f}".format(delta_def)
    stigma_angle = ctf_params['Azimuth of Astigmatism'].values[0]
    stigma_angle_s = "{:.1f}".format(stigma_angle)
    name_str = str(tiff_file)
    
    if scope == 3:
        num_tiff = name_str[-14:-8]
    else:
        num_tiff = name_str[-8:-4]
    
    new_stigma_y, new_stigma_x = calculate_stigma(defocus_u, defocus_v, stigma_angle, scope)



    # Write stigma result to file
    stigma_file = stigma_dir / f"{num_tiff}_{defocus_u_s}_{defocus_v_s}_{delta_def_s}_{stigma_angle_s}+X_{new_stigma_x}_Y_{new_stigma_y}.txt"
    with open(stigma_file, 'w') as file:
        file.write(f"# Columns: #1 - new stigma x; #2 - new stigma y\n")
        file.write(f"{new_stigma_x} {new_stigma_y}\n")
    #generate done flag
    flag_file = flag_dir / f"{inputfile}.done"
    Path(flag_file).touch()

def main(args):
    tiff_files = args.tiff_files
    # tiff_files = ["/home/pp/pptest/20241021_wangtian_0500.tif","/home/pp/pptest/20241021_wangtian_0501.tif","/home/pp/pptest/20241021_wangtian_0502.tif","/home/pp/pptest/20241021_wangtian_0503.tif"]
    gain_out = Path(args.gain_out)
    motioncor2_dir = Path(args.motioncor2_dir)
    ctffind5_dir = Path(args.ctffind5_dir)
    stigma_dir = Path(args.stigma_dir)
    frame_num = args.frame_num
    scope = args.scope_id
    flag_dir = Path(args.flag_dir)

    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(
            process_tiff_file,
            [(tiff_file, gain_out, motioncor2_dir, ctffind5_dir, stigma_dir, args, frame_num, gpu_id, scope, flag_dir)
             for tiff_file, gpu_id in zip(tiff_files, range(len(tiff_files)))]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff_files', nargs='+', help='List of tiff files to process', required=True)
    parser.add_argument('--gain_out', type=str, help='Gain reference output path', required=True)
    parser.add_argument('--binning', type=int, default=1, help='Binning factor', required=True)
    parser.add_argument('--patch', type=int, default=5, help='Patch size')
    parser.add_argument('--dose', type=float, help='Dose per frame', required=True)
    parser.add_argument('--pixel_size', type=float, help='Pixel size', required=True)
    parser.add_argument('--accel_kv', type=float, default=300, help='Acceleration voltage in kV')
    parser.add_argument('--cs_mm', type=float, default=2.7, help='Spherical aberration in mm')
    parser.add_argument('--amp_contrast', type=float, default=0.07, help='Amplitude contrast')
    parser.add_argument('--spectrum_size', type=int, default=512, help='Spectrum size')
    parser.add_argument('--min_res', type=float, default=30.0, help='Minimum resolution')
    parser.add_argument('--max_res', type=float, default=5.0, help='Maximum resolution')
    parser.add_argument('--min_defocus', type=float, default=5000.0, help='Minimum defocus')
    parser.add_argument('--max_defocus', type=float, default=50000.0, help='Maximum defocus')
    parser.add_argument('--defocus_step', type=float, default=100.0, help='Defocus step')
    parser.add_argument('--frame_num', type=int, help='Number of frames in the movie', required=True)
    parser.add_argument('--motioncor2_dir', type=str, help='Directory for MotionCor2 output', required=True)
    parser.add_argument('--ctffind5_dir', type=str, help='Directory for ctffind5 output', required=True)
    parser.add_argument('--stigma_dir', type=str, help='Directory for stigma output', required=True)
    parser.add_argument('--scope_id', type=int, help='BioEM facility microscope number', required=True)
    parser.add_argument('--flag_dir', type=str, help='Directory for done-flag', required=True)
    parser.add_argument("-esamp", "--eer_sampling", type=float, default=2, help="EER sampling mode for MotionCor2")
    parser.add_argument("-efrac", "--eer_fraction", type=float, default=40, help="EER Fractionation for MotionCor2")

    args = parser.parse_args()
    main(args)
