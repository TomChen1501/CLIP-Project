mkdir -p embeddings Resource

echo "Installing gdown..."
pip install gdown

echo "Downloading encoded_tensors.pt..."
gdown --id 1Apj_3U8aEXQqr_2dBoE_TAJhzaB0vaY0 -O embeddings/encoded_tensors.pt

echo "Downloading all_image_embeddings.pt..."
gdown --id 15z6Ah0EcbB_d6YTLaemYo8GHwrQPcNie -O embeddings/all_image_embeddings.pt

echo "Downloading list_attr_celeba.txt..."
gdown --id 1FyDxSKdqfc3zbamWMyZxalGTpLZ70kfh -O Resource/list_attr_celeba.txt

echo "Downloading img_align_celeba.zip..."
gdown --id 1QoCujOf6xTGtXgasCZ_Fcp8e5tXLPMsA -O Resource/img_align_celeba.zip

echo "Unzipping img_align_celeba.zip..."
unzip -o Resource/img_align_celeba.zip -d Resource/
