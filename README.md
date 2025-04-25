# Project Structure

## *First* download_data.py -- Gives us soccernet_data

## pretrainedFasterRCNNExperiments
- Contains baseline benchmarks with various score thresholds
- Contains finetuned model benchmarks with various score thresholds
- Contains sample images from multiple runs

## pretrainedYOLOExperiments
- Contains baseline benchmarks with various score thresholds
- Contains finetuned model training with ROBOFLOW Data, and soccernet data
-   `python process_soccernet.py [soccernet_data dir] ` transforms soccernet_data into YOLO format for finetuning with SoccerNet data
- condense_yolo_soccernet -- eliminates 20 all but 20 frames from yolo format data for quicker benchmarking
- Contains sample images from multiple runs
- `FinetunedYOLORun` contains figures from benchmarking

## fasterRCNNArchitectureExperiments

## customFasterRCNN
