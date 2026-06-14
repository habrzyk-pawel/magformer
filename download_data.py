"""Download Google Speech Commands v0.02 dataset used by speech_demo.py."""
import urllib.request
import tarfile
import os

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE = "speech_commands_v0.02.tar.gz"
EXTRACT_DIR = "SpeechCommands/speech_commands_v0.02"


def main():
    os.makedirs(os.path.dirname(EXTRACT_DIR), exist_ok=True)

    if not os.path.exists(ARCHIVE):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, ARCHIVE)
        print("Download complete.")
    else:
        print(f"{ARCHIVE} already exists, skipping download.")

    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting {ARCHIVE} ...")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print(f"{EXTRACT_DIR} already exists, skipping extraction.")

    print("Dataset ready. Run: python speech_demo.py")


if __name__ == "__main__":
    main()
