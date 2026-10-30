import os
import tarfile
import urllib.request

# Datasets as packaged by LM-BFF (Gao et al., 2021)
URL = "https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar"
ARCHIVE = "datasets.tar"
DATA_DIR = "./data/glue_data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    print(f"Extracting to {DATA_DIR} ...")
    with tarfile.open(ARCHIVE) as tar:
        tar.extractall(path=DATA_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
