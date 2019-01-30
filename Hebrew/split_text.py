import os

root_dir = '/home/wolf/alexeyp'
orig_text = root_dir + 'ocr_datasets/Hebrew/Dataset/Texts/mishna.txt'

new_dir = orig_text[:-4]
os.makedirs(new_dir,exist_ok=True)

with open(orig_text, 'r') as f:
    all_lines = f.readlines()

num_files = 16
lines_per_file = len(all_lines) // num_files

for i in range(num_files + 1):
    if i*lines_per_file < len(all_lines):
        new_file = os.path.join(new_dir, str(i)+'.txt')
        new_text = all_lines[i*lines_per_file:min((i+1)*lines_per_file, len(all_lines))]
        with open(new_file, 'w') as f:
            f.writelines(new_text)