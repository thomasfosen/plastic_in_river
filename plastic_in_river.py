# from https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

import numpy as np

import datasets
from PIL import Image


_DESCRIPTION = """
    This dataset contains photos of rivers on which there may be waste. The waste items are annotated
    through bounding boxes, and are assigned to one of the 4 following categories: plastic bottle, plastic bag,
    another plastic waste, or non-plastic waste. Note that some photos may not contain any waste.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URL = "https://storage.googleapis.com/kili-datasets-public/plastic-in-river/<VERSION>/"

_URLS = {
    "train_images": f"{_URL}train/images.tar.gz",
    "train_annotations": f"{_URL}train/annotations.tar.gz",
    "validation_images": f"{_URL}validation/images.tar.gz",
    "validation_annotations": f"{_URL}validation/annotations.tar.gz",
    "test_images": f"{_URL}test/images.tar.gz",
    "test_annotations": f"{_URL}test/annotations.tar.gz"
}

class PlasticInRiver(datasets.GeneratorBasedBuilder):
    """Download script for the Plastic In River dataset"""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
                {
                    "image": datasets.Image(),
                    "litter": datasets.Sequence(
                        {
                            "label": datasets.ClassLabel(num_classes=4, names=["PLASTIC_BAG", "PLASTIC_BOTTLE", "OTHER_PLASTIC_WASTE", "NOT_PLASTIC_WASTE"]),
                            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        }
                    )
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        urls = {k: v.replace("<VERSION>", f"v{str(self.VERSION)}") for k, v in _URLS.items()}

        downloaded_files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "image_files": dl_manager.iter_archive(downloaded_files[f"{split_name}_images"]),
                    "annotations_files":  dl_manager.iter_archive(downloaded_files[f"{split_name}_annotations"]),
                    "split": split_name,
                },
            ) for split, split_name in [
                (datasets.Split.TRAIN, "train"),
                (datasets.Split.TEST, "test"),
                (datasets.Split.VALIDATION, "validation"),
                ]
        ]

    def _generate_examples(self, image_files, annotations_files, split):

        for idx, (image_file, annotations_file) in enumerate(zip(image_files, annotations_files)):
            image_array = np.array(Image.open(image_file[1]))
            data = {
                "image": image_array,
                "litter": []
            }

            for l in annotations_file[1].readlines():
                numbers = l.decode("utf-8").split(" ")

                data["litter"].append({
                    "label": int(numbers[0]),
                    "bbox": [float(n) for n in numbers[1:]]
                })

            yield idx, data
