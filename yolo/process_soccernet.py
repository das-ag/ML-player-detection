import os
import shutil
import argparse
import pandas as pd
from PIL import Image
import re
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the class mapping (zero-indexed)
CLASS_MAPPING = {
    'goalkeepers team left': 0,
    'goalkeepers team right': 1,
    'player team left': 2,
    'player team right': 3,
    'referee': 4,
    'ball': 5
}

def parse_gameinfo(file_path):
    """Parses gameinfo.ini to map track ID to class name."""
    track_id_to_class_name = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                match = re.match(r'trackletID_(\d+)=\s*(.*?)(?:;|$)', line)
                if match:
                    track_id = int(match.group(1))
                    class_name = match.group(2).strip()
                    track_id_to_class_name[track_id] = class_name
    except FileNotFoundError:
        logging.error(f"gameinfo.ini not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return None
    return track_id_to_class_name

def get_image_dimensions(image_dir, seq_name):
    """Gets width/height by reading the first image found for the sequence."""
    first_image_path = None
    try:
        image_files = [f for f in os.listdir(image_dir)
                       if f.startswith(seq_name + "_") and f.lower().endswith('.jpg')]
        if not image_files:
            logging.error(f"No image files found for sequence {seq_name} in {image_dir}")
            return None, None

        first_image_path = os.path.join(image_dir, image_files[0])
        with Image.open(first_image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                 logging.error(f"Invalid image dimensions ({width}x{height}) read from {first_image_path}")
                 return None, None
            return width, height
    except FileNotFoundError:
        logging.error(f"Image file not found to get dimensions: {first_image_path or 'path not determined'}")
        return None, None
    except Exception as e:
        logging.error(f"Error reading image dimensions for {seq_name} from {image_dir} (file: {first_image_path}): {e}")
        return None, None

def restructure_sequence(seq_dir, images_dest_dir, labels_original_dest_dir):
    """Moves images and labels for a single sequence."""
    seq_name = os.path.basename(seq_dir)
    img1_dir = os.path.join(seq_dir, 'img1')
    gt_dir = os.path.join(seq_dir, 'gt')
    gt_file_path = os.path.join(gt_dir, 'gt.txt')
    seqinfo_path = os.path.join(seq_dir, 'seqinfo.ini')
    det_dir = os.path.join(seq_dir, 'det')

    # Move images
    if os.path.isdir(img1_dir):
        for img_filename in os.listdir(img1_dir):
            if img_filename.lower().endswith('.jpg'):
                old_img_path = os.path.join(img1_dir, img_filename)
                base_img_filename = os.path.splitext(img_filename)[0]
                new_img_filename = f"{seq_name}_{base_img_filename}.jpg"
                new_img_path = os.path.join(images_dest_dir, new_img_filename)
                try:
                    shutil.move(old_img_path, new_img_path)
                except Exception as e:
                    logging.warning(f"Error moving image '{old_img_path}' to '{new_img_path}': {e}")
        try:
            shutil.rmtree(img1_dir)
        except Exception as e:
             logging.warning(f"Could not remove source image dir {img1_dir}: {e}")

    # Move label file
    if os.path.isfile(gt_file_path):
        new_label_filename = f"{seq_name}_gt.txt"
        new_label_path = os.path.join(labels_original_dest_dir, new_label_filename)
        try:
            shutil.move(gt_file_path, new_label_path)
        except Exception as e:
            logging.warning(f"Error moving label file '{gt_file_path}' to '{new_label_path}': {e}")

    # Clean up sequence directory
    if os.path.isfile(seqinfo_path):
        try:
            os.remove(seqinfo_path)
        except OSError as e:
             logging.warning(f"Could not remove {seqinfo_path}: {e}")
    if os.path.isdir(det_dir):
        try:
            shutil.rmtree(det_dir)
        except OSError as e:
            logging.warning(f"Could not remove {det_dir}: {e}")
    if os.path.isdir(gt_dir):
         try:
            # Try removing as empty first, then force if needed (should be empty)
            os.rmdir(gt_dir)
         except OSError:
             try:
                 shutil.rmtree(gt_dir)
             except Exception as e:
                  logging.warning(f"Could not remove gt dir {gt_dir}: {e}")


def convert_labels_to_yolo(seq_name, gameinfo_path, original_gt_path, new_labels_dir, images_dir):
    """Converts the original GT file for a sequence to YOLO format."""
    trackid_to_classname_map = parse_gameinfo(gameinfo_path)
    if trackid_to_classname_map is None:
        logging.warning(f"Skipping label conversion for {seq_name} due to gameinfo parsing error.")
        return 0

    img_width, img_height = get_image_dimensions(images_dir, seq_name)
    if img_width is None or img_height is None:
        logging.warning(f"Skipping label conversion for {seq_name} due to missing image dimensions.")
        return 0

    if not os.path.isfile(original_gt_path):
        logging.warning(f"Original GT file not found, skipping conversion: {original_gt_path}")
        return 0

    files_created_count = 0
    try:
        gt_df = pd.read_csv(original_gt_path, header=None, index_col=False,
                          names=['frame', 'track_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])

        if gt_df.empty:
            logging.info(f"Original GT file is empty: {original_gt_path}")
            return 0

        grouped = gt_df.groupby('frame')

        for frame_id, group in grouped:
            frame_id_str = f"{frame_id}"
            yolo_filename = f"{seq_name}_{frame_id_str}.txt"
            yolo_filepath = os.path.join(new_labels_dir, yolo_filename)
            frame_annotations = []

            for _, row in group.iterrows():
                track_id = int(row['track_id'])
                bb_left = float(row['bb_left'])
                bb_top = float(row['bb_top'])
                bb_width = float(row['bb_width'])
                bb_height = float(row['bb_height'])

                raw_class_name = trackid_to_classname_map.get(track_id)
                if raw_class_name is None:
                    continue

                class_index = CLASS_MAPPING.get(raw_class_name)
                if class_index is None:
                    # Log unrecognized classes once per run? Or let it pass silently?
                    # logging.debug(f"Unrecognized class '{raw_class_name}' for track ID {track_id} in {seq_name}")
                    continue

                x_center = bb_left + bb_width / 2
                y_center = bb_top + bb_height / 2

                norm_x_center = x_center / img_width
                norm_y_center = y_center / img_height
                norm_width = bb_width / img_width
                norm_height = bb_height / img_height

                norm_x_center = max(0.0, min(1.0, norm_x_center))
                norm_y_center = max(0.0, min(1.0, norm_y_center))
                norm_width = max(0.0, min(1.0, norm_width))
                norm_height = max(0.0, min(1.0, norm_height))

                if norm_width <= 0 or norm_height <= 0:
                     continue

                yolo_line = f"{class_index} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                frame_annotations.append(yolo_line)

            if frame_annotations:
                with open(yolo_filepath, 'w') as yolo_file:
                    yolo_file.write("\n".join(frame_annotations) + "\n")
                files_created_count += 1

    except pd.errors.ParserError as e:
         logging.error(f"Error parsing GT file {original_gt_path} with pandas: {e}. Check format.")
    except Exception as e:
        logging.error(f"Unexpected error processing GT file {original_gt_path}: {e}")

    return files_created_count

def process_split(source_split_dir, output_split_dir):
    """Processes a train or test split."""
    logging.info(f"Processing split: {source_split_dir}")
    if not os.path.isdir(source_split_dir):
        logging.error(f"Source split directory not found: {source_split_dir}")
        return

    images_dest_dir = os.path.join(output_split_dir, "images")
    labels_original_dest_dir = os.path.join(output_split_dir, "labels_original") # Temporary
    labels_dest_dir = os.path.join(output_split_dir, "labels")

    os.makedirs(images_dest_dir, exist_ok=True)
    os.makedirs(labels_original_dest_dir, exist_ok=True)
    os.makedirs(labels_dest_dir, exist_ok=True)

    sequences = [d for d in os.listdir(source_split_dir)
                 if d.startswith('SNMOT-') and os.path.isdir(os.path.join(source_split_dir, d))]

    logging.info(f"Found {len(sequences)} sequences in {os.path.basename(source_split_dir)} split.")

    # 1. Initial Restructure (Copy relevant files)
    for seq_name in sequences:
        seq_source_dir = os.path.join(source_split_dir, seq_name)
        # Create a temporary copy of the sequence dir to work from
        seq_temp_work_dir = os.path.join(output_split_dir, seq_name)
        try:
            shutil.copytree(seq_source_dir, seq_temp_work_dir, dirs_exist_ok=True)
            restructure_sequence(seq_temp_work_dir, images_dest_dir, labels_original_dest_dir)
        except Exception as e:
            logging.error(f"Failed initial restructure/copy for {seq_name}: {e}")
            # Clean up partial copy if it exists
            if os.path.exists(seq_temp_work_dir):
                shutil.rmtree(seq_temp_work_dir)


    # 2. Convert Labels
    total_yolo_files = 0
    processed_sequences_conversion = 0
    # Find sequences by looking for the remaining dirs (which now contain gameinfo.ini)
    # Re-list sequences based on the temporary working directories we created
    work_sequences = [d for d in os.listdir(output_split_dir)
                      if d.startswith('SNMOT-') and os.path.isdir(os.path.join(output_split_dir, d))]

    for seq_name in work_sequences:
        seq_temp_work_dir = os.path.join(output_split_dir, seq_name)
        gameinfo_path = os.path.join(seq_temp_work_dir, 'gameinfo.ini')
        original_gt_filename = f"{seq_name}_gt.txt"
        original_gt_path = os.path.join(labels_original_dest_dir, original_gt_filename)

        if not os.path.isfile(gameinfo_path):
             logging.warning(f"gameinfo.ini missing in working dir for {seq_name}, cannot convert labels.")
             continue

        files_created = convert_labels_to_yolo(seq_name, gameinfo_path, original_gt_path, labels_dest_dir, images_dest_dir)
        total_yolo_files += files_created
        if files_created > 0 or os.path.isfile(original_gt_path): # Count sequence if GT existed, even if empty/error
             processed_sequences_conversion += 1

    logging.info(f"Label conversion for {os.path.basename(source_split_dir)} split: Processed {processed_sequences_conversion} sequences, created {total_yolo_files} YOLO files.")

    # 3. Final Cleanup for the split
    logging.info(f"Cleaning up intermediate files for {os.path.basename(source_split_dir)} split...")
    try:
        shutil.rmtree(labels_original_dest_dir)
    except OSError as e:
        logging.warning(f"Could not remove temporary labels_original dir: {e}")

    # Remove the temporary SNMOT working directories
    for seq_name in work_sequences:
         seq_temp_work_dir = os.path.join(output_split_dir, seq_name)
         if os.path.isdir(seq_temp_work_dir):
              try:
                   shutil.rmtree(seq_temp_work_dir)
              except OSError as e:
                    logging.warning(f"Could not remove temporary sequence dir {seq_temp_work_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert SoccerNet dataset to YOLO format.")
    parser.add_argument("source_dir", help="Path to the original SoccerNet dataset directory (containing train/ and test/).")
    parser.add_argument("-o", "--output_dir_name", default="soccernet_data_yolo",
                        help="Name for the output directory to be created in the current working directory (default: soccernet_data_yolo).")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    source_dir = os.path.join(source_dir, "tracking")
    output_base_dir = os.path.join(os.getcwd(), args.output_dir_name)

    if not os.path.isdir(source_dir):
        logging.critical(f"Source directory not found: {source_dir}")
        return

    if os.path.exists(output_base_dir):
        logging.warning(f"Output directory '{output_base_dir}' already exists. Files might be overwritten.")
    else:
        os.makedirs(output_base_dir)
        logging.info(f"Created output directory: {output_base_dir}")


    for split in ['train', 'test']:
        source_split_dir = os.path.join(source_dir, split)
        output_split_dir = os.path.join(output_base_dir, split)
        process_split(source_split_dir, output_split_dir)

    logging.info(f"Processing complete. YOLO formatted data is in: {output_base_dir}")

if __name__ == "__main__":
    main() 