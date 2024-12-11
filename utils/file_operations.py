import os

# Clean a Folder #
def clean_folder(folder):
    # List all files and subfolders in the folder
    del_list = os.listdir(folder)
    # Iterate through the folder list
    for f in del_list:
        # Construct the full file path
        file_path = os.path.join(folder, f)
        # Delete the file
        if os.path.isfile(file_path):
            os.remove(file_path)

# Write Camera Intrinsic Parameters #
def write_intri_to_file(filename, cameraMatrix, distCoeffs):
    # Open the file and write the parameters
    with open(filename, mode='w', encoding='utf-8') as output_file:
        output_file.write("import numpy as np\n")
        output_file.write("cameraMatrix = np.float32(" + str(cameraMatrix.tolist()) + ')\n')
        output_file.write("distCoeff = np.float32(" + str(distCoeffs.tolist()) + ')')
    print("Intrinsics saved to", filename)