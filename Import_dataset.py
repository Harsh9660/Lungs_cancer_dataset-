import kagglehub

# Download latest version
path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")

print("Path to dataset files:", path)