import os
import shutil
import random

def split_dataset(input_folder, output_train_folder, output_test_folder, train_ratio=0.8):
    # Tạo thư mục đầu ra nếu chúng không tồn tại
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    # Lấy danh sách thư mục con (ronaldo, messi, neymar)
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # Duyệt qua từng thư mục con
    for subfolder in subfolders:
        # Lấy danh sách tệp ảnh trong thư mục con
        images = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Tính toán số lượng ảnh cho tập train
        num_train = int(len(images) * train_ratio)

        # Lấy ngẫu nhiên ảnh cho tập train
        train_images = random.sample(images, num_train)

        # Lấy ảnh còn lại cho tập test
        test_images = list(set(images) - set(train_images))

        # Tạo thư mục con cho tập train
        output_train_subfolder = os.path.join(output_train_folder, os.path.basename(subfolder))
        os.makedirs(output_train_subfolder, exist_ok=True)

        # Tạo thư mục con cho tập test
        output_test_subfolder = os.path.join(output_test_folder, os.path.basename(subfolder))
        os.makedirs(output_test_subfolder, exist_ok=True)

        # Di chuyển ảnh vào thư mục tương ứng
        for train_image in train_images:
            shutil.copy(train_image, output_train_subfolder)

        for test_image in test_images:
            shutil.copy(test_image, output_test_subfolder)

# Sử dụng ví dụ:
input_folder = 'dataset'
output_train_folder = 'dataset_split/train'
output_test_folder = 'dataset_split/test'

split_dataset(input_folder, output_train_folder, output_test_folder, train_ratio=0.8)
