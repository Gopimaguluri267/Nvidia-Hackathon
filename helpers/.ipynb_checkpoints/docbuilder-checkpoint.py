# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import pandas as pd

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
)

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def download_and_convert_dataset(output_dir: str) -> pd.DataFrame:
    """
    Downloads the Law Q&A dataset and processes it into a pandas DataFrame.
    """
    DATASET_URL = "https://huggingface.co/datasets/ymoslem/Law-StackExchange/resolve/main/law-stackexchange-questions-answers.json"
    download_dir = os.path.join(output_dir, "raw")
    os.makedirs(download_dir, exist_ok=True)

    downloader = LawQADownloader(download_dir)
    raw_fp = downloader.download(DATASET_URL)

    iterator = LawQAIterator()
    rows = []

    for record in iterator.iterate(raw_fp):
        rows.append(record)

    return pd.DataFrame(rows)


class LawQADownloader(DocumentDownloader):
    """
    A class for downloading Law QA dataset.
    """

    def __init__(self, download_dir: str):
        super().__init__()

        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        self._download_dir = download_dir
        print("Download directory: ", self._download_dir)

    def download(self, url: str) -> str:
        """
        Downloads the Law QA dataset from the given URL.

        Args:
            url (str): The URL of the Law QA dataset.

        Returns:
            str: The path of the downloaded file.

        """
        filename = os.path.basename(url)
        output_file = os.path.join(self._download_dir, filename)

        if os.path.exists(output_file):
            print(f"File '{output_file}' already exists, skipping download.")
            return output_file

        print(f"Downloading Law QA dataset from '{url}'...")
        response = requests.get(url)

        with open(output_file, "wb") as file:
            file.write(response.content)

        return output_file


class LawQAIterator(DocumentIterator):
    def __init__(self):
        super().__init__()
        self._counter = -1
        self._extractor = LawQAExtractor()

    def iterate(self, file_path):
        self._counter = -1
        file_name = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            json_content = json.load(file)

        for row in json_content:
            self._counter += 1
            extracted_content = self._extractor.extract(row)

            if extracted_content is None:
                continue

            id, content = extracted_content
            meta = {
                "filename": file_name,
                "id": f"law-stackexchange-qa-{id}",
            }

            yield {**meta, **content}


class LawQAExtractor(DocumentExtractor):
    def extract(self, content: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            if not content.get("answers"):
                return None

            id = content["question_id"]
            q_title = content["question_title"]
            q_body = content["question_body"]
            q_score = content["score"]
            tags = ",".join(sorted(content["tags"]))

            best_answer = content["answers"][0]
            best_answer_score = best_answer["score"]
            best_answer_text = best_answer["body"]

            # Clean HTML
            q_title = self._clean_html(q_title)
            q_body = self._clean_html(q_body)
            best_answer_text = self._clean_html(best_answer_text)

            return id, {
                "title": q_title,
                "question": q_body,
                "question_score": q_score,
                "answer": best_answer_text,
                "answer_score": best_answer_score,
                "tags": tags
            }
        except Exception as e:
            print(f"Error processing content: {str(e)}")
            return None



    def _clean_html(self, text: str) -> str:
        text = BeautifulSoup(text, "lxml").get_text()
        return re.sub(r"\s+", " ", text).strip()
