mkdir -p Resource

echo "Current dir: $(pwd)"
ls -lh Resource/

echo "Downloading encoded_tensors.pt..."
gdown 1Apj_3U8aEXQqr_2dBoE_TAJhzaB0vaY0 -O Resource/encoded_tensors.pt
echo "Current dir: $(pwd)"
ls -lh Resource/

echo "Downloading all_image_embeddings.pt..."
gdown 15z6Ah0EcbB_d6YTLaemYo8GHwrQPcNie -O Resource/all_image_embeddings.pt
echo "Current dir: $(pwd)"
ls -lh Resource/

echo "Downloading list_attr_celeba.txt..."
gdown 1FyDxSKdqfc3zbamWMyZxalGTpLZ70kfh -O Resource/list_attr_celeba.txt
echo "Current dir: $(pwd)"
ls -lh Resource/

echo "Downloading img_align_celeba.zip..."
gdown 1QoCujOf6xTGtXgasCZ_Fcp8e5tXLPMsA -O Resource/img_align_celeba.zip
echo "Current dir: $(pwd)"
ls -lh Resource/

echo "Unzipping img_align_celeba.zip..."
unzip -o Resource/img_align_celeba.zip -d Resource/
echo "Current dir: $(pwd)"
ls -lh Resource/