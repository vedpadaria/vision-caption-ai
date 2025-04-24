import json
import os
import nltk
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data as data
from tqdm import tqdm
from vocabulary import Vocabulary


class CoCoDataset(data.Dataset):
    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        
        # Normalize and verify paths
        self.img_folder = os.path.normpath(img_folder)
        annotations_file = os.path.normpath(annotations_file)
        
        # Debug prints to verify paths
        print("\n" + "="*50)
        print(f"Looking for annotations at: {annotations_file}")
        print(f"Looking for images at: {self.img_folder}")
        print("="*50 + "\n")
        
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(
                f"Annotations file not found at: {annotations_file}\n"
                f"Directory contents: {os.listdir(os.path.dirname(annotations_file))}"
            )
            
        if not os.path.exists(self.img_folder):
            raise FileNotFoundError(
                f"Image folder not found at: {self.img_folder}\n"
                f"Parent directory contents: {os.listdir(os.path.dirname(self.img_folder))}"
            )

        # create vocabulary from the captions
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )
        
        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")

            tokenized_captions = [
                nltk.tokenize.word_tokenize(
                    str(self.coco.anns[self.ids[index]]["caption"]).lower()
                )
                for index in tqdm(np.arange(len(self.ids)))
            ]
            self.caption_lengths = [len(token) for token in tokenized_captions]
        else:
            with open(annotations_file, 'r') as f:
                test_info = json.load(f)
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        if self.mode == "train":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            
            img_path = os.path.join(self.img_folder, path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = [self.vocab(self.vocab.start_word)]
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            return image, caption
        else:
            path = self.paths[index]
            img_path = os.path.join(self.img_folder, path)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")

            pil_image = Image.open(img_path).convert("RGB")
            orig_image = np.array(pil_image)
            image = self.transform(pil_image)

            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train":
            return len(self.ids)
        else:
            return len(self.paths)