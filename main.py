from tsx.datasets.base import load_ecg200, load_ecg5000

ds = load_ecg5000(download=True)

print(ds.x_train[0], ds.y_train[0])
