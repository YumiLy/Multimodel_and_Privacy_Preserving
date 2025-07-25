# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Librispeech automatic speech recognition dataset."""

from __future__ import absolute_import, division, print_function

import glob
import os

import datasets


_CITATION = """\
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.
Note that in order to limit the required storage for preparing this dataset, the audio
is stored in the .flac format and is not converted to a float32 array. To convert, the audio
file to a float32 array, please make use of the `.map()` function as follows:
```python
import soundfile as sf
def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch
dataset = dataset.map(map_to_array, remove_columns=["file"])
```
"""

_URL = "http://www.openslr.org/12"
# _DL_URL = "https://s3.amazonaws.com/datasets.huggingface.co/librispeech_asr/2.1.0/"
# _DL_URL = "https://s3.amazonaws.com/datasets.huggingface.co/librispeech_asr/2.1.0/"

# _DL_URLS = {
#     "clean": {
#         "dev": _DL_URL + "dev_clean.tar.gz",
#         "test": _DL_URL + "test_clean.tar.gz",
#         "train": _DL_URL + "train-clean-100.tar.gz",
#     }
# }


class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(LibrispeechASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class LibrispeechASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    BUILDER_CONFIGS = [
        LibrispeechASRConfig(name="clean", description="'Clean' speech."),
        LibrispeechASRConfig(name="other", description="'Other', more challenging, speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                }
            ),
            supervised_keys=("speech", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # archive_path = dl_manager.download_and_extract(_DL_URLS[self.config.name])
        archive_path = {
        "dev": "/home/lai/datasets/Librispeech/dev_clean",
        "test": "/home/lai/datasets/Librispeech/test_clean",
        "train": "/home/lai/datasets/Librispeech/train_clean100"
        }
        # return [
        #     datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"archive_path": archive_path["dev"], "split_name": f"dev_{self.config.name}"}),
        # ]

        self.speaker_dict = self._load_speaker_metadata("/home/lai/datasets/Librispeech/SPEAKERS.TXT")

        return [
            datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={"archive_path": archive_path["dev"], "split_name": "dev-clean"},
            ),
            datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={"archive_path": archive_path["test"], "split_name": "test-clean"},
            ),
            datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={"archive_path": archive_path["train"], "split_name": "train-clean-100"},
            ),
        ]


    def _generate_examples(self, archive_path, split_name):
        """Generate examples from a Librispeech archive_path."""
        transcripts_glob = os.path.join(archive_path, "*/*/*.txt")
        for transcript_file in sorted(glob.glob(transcripts_glob)):
            path = os.path.dirname(transcript_file)
            # with open(os.path.join(path, transcript_file)) as f:
            with open(transcript_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    key, transcript = line.split(" ", 1)
                    audio_file = f"{key}.flac"
                    speaker_id, chapter_id = [int(el) for el in key.split("-")[:2]]
                    gender = self.speaker_dict.get(speaker_id, "unknown")
                    example = {
                        "id": key,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "file": os.path.join(path, audio_file),
                        "audio": os.path.join(path, audio_file),
                        "text": transcript,
                        "gender": gender,
                    }
                    yield key, example
    
    def _load_speaker_metadata(self, speaker_file):
        speaker_dict = {}
        with open(speaker_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(";") or not line:
                    continue
                parts = [x.strip() for x in line.split("|")]
                if len(parts) >= 2:
                    try:
                        speaker_id = int(parts[0])
                        gender = parts[1]
                        speaker_dict[speaker_id] = gender
                    except ValueError:
                        continue  # Skip header or bad lines
        return speaker_dict