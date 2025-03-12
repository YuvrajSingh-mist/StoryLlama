import gdown
import os

file_id = "1Vmdl5a41TyOWy2gI_axpZemGJhHjw-ZK"
url = f"https://drive.google.com/uc?id={file_id}"

os.makedirs('weights/', exist_ok=True)

output = "weights/snapshot.pt"
gdown.download(url, output, quiet=False)

print("Model downloaded successfully!")