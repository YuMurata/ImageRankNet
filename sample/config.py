from pathlib import Path

tfrecords_dir = Path(__file__).parent/'tfrecords'
tfrecords_dir.mkdir(exist_ok=True, parents=True)

logs_dir = Path(__file__).parent/'logs'
logs_dir.mkdir(exist_ok=True, parents=True)

weights_dir = Path(__file__).parent/'weights'
weights_dir.mkdir(exist_ok=True, parents=True)

IMAGE_SHAPE = (10, 10, 3)
