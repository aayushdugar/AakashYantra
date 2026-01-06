import subprocess
import time
import os
import sys
 
# ==========================
# CONFIGURATION
# ==========================
PROJECT_DIR = "/home/ay/Downloads/AakashYantra-main"
VENV_ACTIVATE = "/home/ay/Downloads/AakashYantra-main/venv/bin/activate" 
 
CAPTURE_SCRIPT = "capture_images.py"
VARI_SCRIPT = "apply_vari.py"
HEATMAP_SCRIPT = "map_generator.py"
 
BOOT_DELAY_SECONDS = 6  # wait after power-on
 
# ==========================
# HELPER FUNCTION
# ==========================
def run_script(script_name):

    subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        check=True
    )
 
# ==========================
# MAIN PIPELINE
# ==========================
def main():
    print(" AakashYantra Master Pipeline Started")
 
    # 1. Wait for system to stabilize after boot
    print(f" Waiting {BOOT_DELAY_SECONDS} seconds for system boot...")
    time.sleep(BOOT_DELAY_SECONDS)
 
    # 2. Change directory to project
    print("?? Changing to project directory")
    os.chdir(PROJECT_DIR)
 
    # Safety check
    if not os.path.exists(VENV_ACTIVATE):
        print("? Virtual environment not found")
        sys.exit(1)
 
    try:
        # 3. Capture Images
        print("?? Starting image capture...")
        run_script(CAPTURE_SCRIPT)
 
        # 4. VARI Processing
        print("?? Running VARI calculations...")
        run_script(VARI_SCRIPT)
 
        # 5. Heatmap Generation
        print("?? Generating heatmap...")
        run_script(HEATMAP_SCRIPT)
 
        print("? Pipeline completed successfully")
 
    except subprocess.CalledProcessError as e:
        print("? Pipeline failed:", e)
        sys.exit(1)
 
# ==========================
if __name__ == "__main__":
    main()