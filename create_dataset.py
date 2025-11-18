import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files("jessicali9530/celeba-dataset", path='dataset/', unzip=True)