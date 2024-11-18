import os
import shutil
from sklearn.model_selection import train_test_split
from src.cnnClassifier import logger
from src.cnnClassifier.entity import DataSplitConfig

class DataSplitter:
    def __init__(self, config: DataSplitConfig):
        self.config = config

    def split_data(self, source_dir):
        classes = os.listdir(source_dir)  # Các thư mục con (class)
        train_dir = self.config.train_dir
        val_dir = self.config.val_dir
        test_dir = self.config.test_dir

        for class_name in classes:
            class_path = os.path.join(source_dir, class_name)
            if os.path.isdir(class_path):  # Kiểm tra nếu là thư mục
                files = os.listdir(class_path)  # Lấy danh sách file trong lớp
                if len(files) < 2:  # Nếu số file quá ít, bỏ qua
                    logger.warning(f"Lớp {class_name} có ít hơn 2 file, bỏ qua chia dữ liệu.")
                    continue

                # Chia dữ liệu thành train, val, test
                train_files, temp_files = train_test_split(
                    files, test_size=1 - self.config.split_ratios[0], random_state=42
                )
                val_files, test_files = train_test_split(
                    temp_files,
                    test_size=self.config.split_ratios[2] / (self.config.split_ratios[1] + self.config.split_ratios[2]),
                    random_state=42
                )

                # Copy file vào các thư mục tương ứng
                self._copy_files(class_path, train_files, os.path.join(train_dir, class_name))
                self._copy_files(class_path, val_files, os.path.join(val_dir, class_name))
                self._copy_files(class_path, test_files, os.path.join(test_dir, class_name))

    @staticmethod
    def _copy_files(source_class_dir, files, dest_class_dir):
        os.makedirs(dest_class_dir, exist_ok=True)  # Tạo thư mục đích nếu chưa tồn tại
        for file in files:
            src_path = os.path.join(source_class_dir, file)
            dest_path = os.path.join(dest_class_dir, file)
            shutil.copy(src_path, dest_path)  # Sao chép file